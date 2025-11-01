# app.py
# FastAPI para servir o modelo (completação de solo_elevacao e declive_graus)
# - Elevação via Google Elevation API
# - Declive via raster em data/slope_srtm_sp.tif
# - Normalização do payload e coerência de "intensidade_chuva"
# - LOGS estruturados (JSON) do que entra no modelo e da resposta
# - Pós-ajuste: redutor por declive e teto para cenários sem chuva (configuráveis por env)

# >>>>>>> PATCH CRÍTICO PARA DES-SERIALIZAR O MODELO <<<<<<<
from model_bundle import PreprocessXGB   # garante classe carregada de um módulo real
import sys, model_bundle
# Quando o pickle pedir "__main__.PreprocessXGB" (ou "__mp_main__"), aponte para model_bundle
sys.modules['__main__'] = model_bundle
sys.modules['__mp_main__'] = model_bundle
# >>>>>>> FIM DO PATCH <<<<<<<

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from functools import lru_cache
from pathlib import Path
import os
import traceback

import joblib
import pandas as pd
import numpy as np
import httpx
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as rio_transform

# ===== LOGGING =====
import logging, time, json
logger = logging.getLogger("alagouai")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)  # troque para DEBUG se quiser mais verbosidade

# =========================
# Caminhos
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"
DEFAULT_MODEL_NAME = "modelo_xgboost_novo.pkl"  # ajuste para o nome real do seu .pkl
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME
SLOPE_TIF_PATH = DATA_DIR / "slope_srtm_sp.tif"

# =========================
# Config externas
# =========================
# Recom.: use variável de ambiente; aqui há um fallback apenas para dev.
GOOGLE_MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY", "AIzaSyCTOi-ejXpzRg_rNa9zrlFNSxRCIHcqb_8").strip()
GOOGLE_ELEVATION_URL = "https://maps.googleapis.com/maps/api/elevation/json"
HTTP_TIMEOUT_S = 6.0

# =========================
# App
# =========================
app = FastAPI(
    title="AlagouAI Predictor API",
    version="2.1.0",
    description="Serve o modelo XGBoost e completa features de elevação e declive, com pós-ajustes de risco."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ajuste para os domínios do app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Middleware de request/latência (logs) =====
@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    resp = await call_next(request)
    dur_ms = int((time.time() - start) * 1000)
    try:
        logger.info(json.dumps({
            "event": "http_request",
            "path": str(request.url.path),
            "method": request.method,
            "status": resp.status_code,
            "dur_ms": dur_ms
        }, ensure_ascii=False))
    except Exception:
        pass
    return resp

# =========================
# Schemas
# =========================
class PredictRequest(BaseModel):
    latitude: float = Field(..., description="Latitude WGS84")
    longitude: float = Field(..., description="Longitude WGS84")
    temperatura: float
    umidade: float
    pressao: float
    precipitacaoChuva: float
    pontoOrvalho: float
    tempoChuva: int
    precipitacaoAcumulada: float
    intensidadeChuva: str
    # Campo opcional para sobrepor cálculo automático (se quiser enviar pronto)
    soloElevacao: Optional[float] = None
    decliveGraus: Optional[float] = None

class PredictResponse(BaseModel):
    risk: float
    chance: str
    bucket: str
    model: str

# =========================
# Recursos globais (modelo e raster)
# =========================
_model = None
_loaded_path: Optional[Path] = None
_slope_ds: Optional[rasterio.io.DatasetReader] = None
_slope_crs_epsg4326 = "EPSG:4326"

def _try_load_model(path: Path):
    global _model, _loaded_path
    try:
        obj = joblib.load(path)
        # Se vier como dict bundle, reconstrói o wrapper local:
        if isinstance(obj, dict) and {"preprocessor","xgb","feature_order"} <= obj.keys():
            _model = PreprocessXGB(
                preprocessor=obj["preprocessor"],
                xgb=obj["xgb"],
                feature_order=obj.get("feature_order", [])
            )
        else:
            _model = obj
        _loaded_path = path
        print(f"[OK] Modelo carregado: {path}")
        logger.info(json.dumps({
            "event":"model_loaded",
            "path": str(path),
            "model_type": type(_model).__name__,
            "has_feature_order": bool(getattr(_model, "feature_order", None))
        }, ensure_ascii=False))
    except Exception:
        _model, _loaded_path = None, None
        print("[ERRO] Falha ao carregar modelo:")
        traceback.print_exc()
        logger.exception("model_load_failed")

def _pick_latest_pkl(models_dir: Path) -> Optional[Path]:
    cands = list(models_dir.glob("*.pkl"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

def _open_slope():
    global _slope_ds
    if SLOPE_TIF_PATH.exists():
        try:
            _slope_ds = rasterio.open(SLOPE_TIF_PATH)
            print(f"[OK] Raster de declive aberto: {SLOPE_TIF_PATH}")
            logger.info(json.dumps({
                "event":"slope_opened",
                "path": str(SLOPE_TIF_PATH),
                "crs": str(_slope_ds.crs) if _slope_ds and _slope_ds.crs else None
            }, ensure_ascii=False))
        except Exception:
            _slope_ds = None
            print("[AVISO] Não foi possível abrir o raster de declive:")
            traceback.print_exc()
            logger.exception("slope_open_failed")
    else:
        msg = f"Raster de declive não encontrado: {SLOPE_TIF_PATH}"
        print(f"[AVISO] {msg}")
        logger.warning(json.dumps({"event":"slope_missing","path":str(SLOPE_TIF_PATH)}, ensure_ascii=False))

@app.on_event("startup")
def _startup():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if DEFAULT_MODEL_PATH.exists():
        _try_load_model(DEFAULT_MODEL_PATH)
    else:
        latest = _pick_latest_pkl(MODELS_DIR)
        if latest:
            _try_load_model(latest)
        else:
            msg = f"Nenhum .pkl encontrado em {MODELS_DIR}"
            print(f"[AVISO] {msg}")
            logger.warning(json.dumps({"event":"no_model_found","dir":str(MODELS_DIR)}, ensure_ascii=False))
    _open_slope()

# =========================
# Utils de features
# =========================
INT_MAP = {
    "sem chuva": "sem chuva",
    "chuva fraca": "chuva fraca",
    "chuva moderada": "chuva moderada",
    "chuva forte": "chuva forte",
    "céu limpo": "sem chuva",
    "ceu limpo": "sem chuva",
    "nublado": "sem chuva",
    "trovoada": "chuva moderada",
    "trovoada com chuva": "chuva forte",
    "trovoada com chuva forte": "chuva forte",
}

def _format_percent_br(p: float, decimals: int = 2) -> str:
    return f"{p*100:.{decimals}f}".replace(".", ",") + "%"

def _classificar_risco(p: float) -> str:
    if p <= 0.30:
        return "baixo"
    elif p <= 0.70:
        return "medio"
    else:
        return "alto"

def _normalize_intensity(txt: str) -> str:
    s = str(txt or "").strip().lower()
    return INT_MAP.get(s, "sem chuva")

@lru_cache(maxsize=10000)
def _cached_elevation(lat_6: float, lon_6: float) -> Optional[float]:
    """Chama Google Elevation; cache por coord arredondada a 6 casas."""
    if not GOOGLE_MAPS_KEY:
        return None
    params = {"locations": f"{lat_6:.6f},{lon_6:.6f}", "key": GOOGLE_MAPS_KEY}
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT_S) as client:
            r = client.get(GOOGLE_ELEVATION_URL, params=params)
            r.raise_for_status()
            js = r.json()
            if js.get("status") == "OK" and js.get("results"):
                return float(js["results"][0]["elevation"])
    except Exception:
        # Não derruba a API por falha externa
        logger.warning(json.dumps({
            "event":"elevation_fetch_failed",
            "lat": lat_6, "lon": lon_6
        }, ensure_ascii=False))
    return None

def _sample_slope(lat: float, lon: float) -> Optional[float]:
    """Amostra declive (em graus) no raster. Assume que o raster já é slope (graus)."""
    if _slope_ds is None:
        return None
    x, y = lon, lat
    try:
        # reprojeta se necessário
        if _slope_ds.crs and _slope_ds.crs.to_string().upper() != _slope_crs_epsg4326:
            xs, ys = rio_transform(_slope_crs_epsg4326, _slope_ds.crs, [lon], [lat])
            x, y = xs[0], ys[0]
        row, col = _slope_ds.index(x, y)
        band = _slope_ds.read(1, out_dtype="float32", resampling=Resampling.bilinear)
        val = float(band[row, col])
        # trata NoData
        nodata = _slope_ds.nodata
        if nodata is not None and np.isclose(val, nodata):
            return None
        if np.isnan(val) or np.isinf(val):
            return None
        # sanity: slope não-negativa
        return max(0.0, val)
    except Exception:
        logger.exception("slope_sample_failed")
        return None

def _complete_features(req: PredictRequest) -> Dict[str, Any]:
    """Monta dict snake_case pronto pro modelo, completando solo_elevacao / declive_graus se faltarem."""
    data = {
        "temperatura": float(req.temperatura),
        "umidade": float(req.umidade),
        "pressao": float(req.pressao),
        "precipitacao_chuva": float(req.precipitacaoChuva),
        "ponto_orvalho": float(req.pontoOrvalho),
        "tempo_chuva": int(req.tempoChuva),
        "precipitacao_acumulada": float(req.precipitacaoAcumulada),
        "intensidade_chuva": _normalize_intensity(req.intensidadeChuva),
    }

    # coerência: se não choveu nas métricas, zera intensidade textual
    if data["precipitacao_chuva"] == 0.0 and data["tempo_chuva"] == 0:
        data["intensidade_chuva"] = "sem chuva"

    # solo_elevacao: usa valor enviado ou consulta Google
    if req.soloElevacao is not None:
        data["solo_elevacao"] = float(req.soloElevacao)
    else:
        elev = _cached_elevation(round(req.latitude, 6), round(req.longitude, 6))
        if elev is not None:
            data["solo_elevacao"] = float(elev)

    # declive_graus: usa valor enviado ou amostra raster
    if req.decliveGraus is not None:
        data["declive_graus"] = max(0.0, float(req.decliveGraus))
    else:
        slope = _sample_slope(req.latitude, req.longitude)
        if slope is not None:
            data["declive_graus"] = float(slope)

    # Garante chaves mesmo sem fonte externa (imputer do modelo trata NaN)
    if "solo_elevacao" not in data:
        data["solo_elevacao"] = np.nan
    if "declive_graus" not in data:
        data["declive_graus"] = np.nan

    return data

# --- Redutor por declive (quanto maior o declive, menor a prob) ---
def slope_damping(slope_deg: Optional[float]) -> float:
    """
    Retorna um fator ∈ (0,1] que só reduz a probabilidade para declives maiores.
    k controla a sensibilidade: 0.04–0.10 são valores razoáveis.
    Pode ser ajustado via env SLOPE_DAMP_K (default: 0.06).
    """
    if slope_deg is None:
        return 1.0
    try:
        s = float(slope_deg)
    except Exception:
        return 1.0
    if np.isnan(s):
        return 1.0
    k = float(os.getenv("SLOPE_DAMP_K", "0.06"))
    s = max(0.0, s)
    return float(np.exp(-k * s))

def apply_no_rain_cap(proba: float, feats: Dict[str, Any]) -> float:
    """
    Se não há chuva (0 mm, 0 min, 'sem chuva'), aplica um teto (cap) na prob.
    Ajustável via env NO_RAIN_CAP (default: 0.25).
    """
    p = float(proba)
    try:
        ch = float(feats.get("precipitacao_chuva", 0.0))
        tc = float(feats.get("tempo_chuva", 0))
        it = str(feats.get("intensidade_chuva", "")).strip().lower()
    except Exception:
        return p
    if ch == 0.0 and tc == 0 and it == "sem chuva":
        cap = float(os.getenv("NO_RAIN_CAP", "0.25"))
        return min(p, cap)
    return p

def _df_for_model(features: Dict[str, Any]) -> pd.DataFrame:
    """
    DataFrame exatamente com as colunas esperadas pelo modelo.
    - Se wrapper PreprocessXGB: usa .feature_order e preenche NaN para faltantes.
    - Se Pipeline: usa ordem fallback.
    Faz log estruturado do que está indo pro modelo.
    """
    def _build(order: list[str]) -> pd.DataFrame:
        # linha base com NaN
        row = {c: np.nan for c in order}
        # sobrescreve com o que chegou
        for k, v in features.items():
            if k in row:
                row[k] = v
        df = pd.DataFrame([row], columns=order)

        # logging: faltantes e extras
        try:
            missing = [c for c in order if pd.isna(df.iloc[0][c])]
            extras  = [k for k in features.keys() if k not in order]
            logger.info(json.dumps({
                "event": "model_input",
                "expected_cols": order,
                "missing_filled_with_NaN": missing,
                "extra_ignored": extras,
                "dtypes": {c: str(df[c].dtype) for c in df.columns}
            }, ensure_ascii=False))
        except Exception:
            pass

        return df

    if hasattr(_model, "feature_order") and isinstance(_model.feature_order, (list, tuple)):
        return _build(list(_model.feature_order))

    fallback = [
        "temperatura","umidade","pressao","precipitacao_chuva","ponto_orvalho",
        "tempo_chuva","precipitacao_acumulada","declive_graus","solo_elevacao","intensidade_chuva"
    ]
    return _build(fallback)

# =========================
# Rotas
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "degraded",
        "model_loaded": _model is not None,
        "model_path": str(_loaded_path) if _loaded_path else str(DEFAULT_MODEL_PATH),
        "slope_loaded": bool(_slope_ds is not None),
        "slope_path": str(SLOPE_TIF_PATH),
        "uses_google": bool(GOOGLE_MAPS_KEY),
    }

@app.get("/_model_debug")
def model_debug():
    fo = getattr(_model, "feature_order", None)
    return {
        "model_type": type(_model).__name__,
        "feature_order": fo,
        "n_features_expected": len(fo) if isinstance(fo, (list, tuple)) else None,
        "model_path": str(_loaded_path) if _loaded_path else None
    }

@app.post("/_features")
def debug_features(req: PredictRequest):
    feats = _complete_features(req)
    X = _df_for_model(feats)
    return {
        "features_final": feats,
        "columns": list(X.columns),
        "dtypes": {c: str(X[c].dtype) for c in X.columns},
        "row": X.iloc[0].to_dict()
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    feats = _complete_features(req)

    # Log seguro das features de entrada (sem headers/keys)
    try:
        logger.info(json.dumps({"event":"features_built","features":feats}, ensure_ascii=False))
    except Exception:
        pass

    X = _df_for_model(feats)

    # Predizer (probabilidade "crua" do modelo)
    try:
        proba_raw = float(_model.predict_proba(X)[:, 1][0])
    except Exception as e:
        logger.exception("predict_proba_failed")
        raise HTTPException(status_code=400, detail=f"Erro durante a predição: {e}")

    # 1) Redução por declive (apenas reduz; nunca aumenta)
    declive = feats.get("declive_graus", None)
    factor_slope = slope_damping(declive)
    proba = proba_raw * factor_slope

    # 2) Teto em cenário sem chuva (opcional; default ON via env)
    proba = apply_no_rain_cap(proba, feats)

    # Clamps finais por segurança
    proba = float(np.clip(proba, 0.0, 1.0))

    bucket = _classificar_risco(proba)
    resp = {
        "risk": round(proba, 2),
        "chance": _format_percent_br(proba, 2),
        "bucket": bucket,
        "model": "xgb"
    }

    # Loga tudo: prob crua, declive, fator e prob final
    try:
        logger.info(json.dumps({
            "event":"predict_done",
            "proba_raw": round(proba_raw, 6),
            "declive_graus": declive,
            "slope_factor": round(factor_slope, 6),
            "proba_final": round(proba, 6),
            "response": resp
        }, ensure_ascii=False))
    except Exception:
        pass

    return PredictResponse(**resp)

# Como executar local:
# uvicorn app:app --host 0.0.0.0 --port 8000

