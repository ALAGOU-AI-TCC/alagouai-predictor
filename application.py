# application.py
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
import os
import warnings

import pandas as pd
import numpy as np
from joblib import load as joblib_load
from sklearn.exceptions import InconsistentVersionWarning

# HTTP p/ Google Elevation
import requests

# Raster (declive)
import rasterio
from rasterio.warp import transform as rio_transform

# =======================
# Config / Paths
# =======================
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "modelo_xgboost.pkl")

# GeoTIFF de declive em graus; idealmente EPSG:4326
SLOPE_TIF = os.getenv("SLOPE_TIF", "data/slope_srtm_sp.tif")

# Chave da Google Elevation API (defina no ambiente, se possível)
GOOGLE_MAPS_KEY = "AIzaSyCTOi-ejXpzRg_rNa9zrlFNSxRCIHcqb_8"
# GOOGLE_MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY")

# Bins iguais aos usados no treino
DECLIVE_BINS  = [0, 2, 4, 6, 8, 10, 15, 60]
DECLIVE_LABEL = [f"{DECLIVE_BINS[i]}–{DECLIVE_BINS[i+1]}°" for i in range(len(DECLIVE_BINS)-1)]

# =======================
# App
# =======================
app = FastAPI(title="AlagouAI Predictor", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =======================
# Carregamentos
# =======================
xgb_pipeline = joblib_load(MODEL_PATH)
if not hasattr(xgb_pipeline, "predict_proba"):
    raise RuntimeError("modelo_xgboost.pkl não é um Pipeline com predict_proba.")

slope_ds = rasterio.open(SLOPE_TIF)  # mantém o raster aberto

# =======================
# Helpers
# =======================
def round2(v: float) -> float:
    return float(Decimal(v).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def fmt_ptbr_2dec(v: float) -> str:
    return f"{round2(v):.2f}".replace(".", ",")

def bucketize(p: float) -> str:
    if p <= 0.30: return "baixo"
    if p <= 0.70: return "medio"
    return "alto"

def sample_raster(ds: rasterio.DatasetReader, lon: float, lat: float) -> Optional[float]:
    """Amostra valor do raster em lon/lat, independente do CRS do raster."""
    try:
        if ds.crs and ds.crs.to_epsg() != 4326:
            x, y = rio_transform("EPSG:4326", ds.crs, [lon], [lat])
            x, y = x[0], y[0]
        else:
            x, y = lon, lat
        row, col = ds.index(x, y)
        if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
            return None
        val = ds.read(1)[row, col]
        if ds.nodata is not None and (val == ds.nodata):
            return None
        if np.isnan(val):
            return None
        return float(val)
    except Exception:
        return None

def get_google_elevation(lat: float, lon: float, timeout: float = 3.5) -> Optional[float]:
    """
    Consulta a Google Elevation API e retorna a elevação em metros.
    Retorna None em qualquer falha (HTTP, quota, sem resultados, chave ausente etc.).
    """
    if not GOOGLE_MAPS_KEY:
        return None
    try:
        url = "https://maps.googleapis.com/maps/api/elevation/json"
        params = {"locations": f"{lat},{lon}", "key": GOOGLE_MAPS_KEY}
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "OK":
            return None
        results = data.get("results", [])
        if not results:
            return None
        elev_m = results[0].get("elevation")
        if elev_m is None:
            return None
        return float(elev_m)
    except Exception:
        return None

# =======================
# Schemas
# =======================
class Input(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    latitude:  float = Field(..., alias="latitude")
    longitude: float = Field(..., alias="longitude")

    temperatura: float = Field(..., alias="temperatura")
    umidade: float = Field(..., alias="umidade")
    pressao: float = Field(..., alias="pressao")

    precipitacao_chuva: float = Field(..., alias="precipitacaoChuva")
    ponto_orvalho: float = Field(..., alias="pontoOrvalho")
    tempo_chuva: int = Field(..., alias="tempoChuva")
    precipitacao_acumulada: float = Field(..., alias="precipitacaoAcumulada")
    intensidade_chuva: str = Field(..., alias="intensidadeChuva")

class PredictResponse(BaseModel):
    risk: float      # 0..1 (2 casas)
    chance: str      # "40,20%"
    bucket: str      # baixo/medio/alto
    model: str
    meta: dict

# =======================
# Endpoints
# =======================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(inp: Input):
    p = inp.model_dump(by_alias=False)

    # 1) Declive (graus) via raster
    declive_graus = sample_raster(slope_ds, p["longitude"], p["latitude"])
    if declive_graus is None:
        raise HTTPException(status_code=422, detail="Declive não encontrado para essa coordenada.")

    # 1.1) Elevação via Google (metros) — fallback p/ 0.0 se falhar
    elev = get_google_elevation(p["latitude"], p["longitude"])
    elev_ok = elev is not None
    solo_elevacao = elev if elev_ok else 0.0

    # 2) Clippings / features derivadas (iguais ao treino; sem chuva_media_h)
    tempo_chuva = int(np.clip(p["tempo_chuva"], 0, 4))
    precipitacao_chuva = max(0.0, float(p["precipitacao_chuva"]))
    precipitacao_acumulada = max(0.0, float(p["precipitacao_acumulada"]))

    slope_plano = 1 if declive_graus < 2 else 0
    declive_bin = pd.cut(pd.Series([declive_graus]),
                         bins=DECLIVE_BINS, labels=DECLIVE_LABEL,
                         include_lowest=True, right=False).astype(str).iloc[0]
    intensidade_chuva = str(p["intensidade_chuva"]).strip().lower()

    # 3) Monta DataFrame exatamente com as colunas do treino
    row = {
        "temperatura": float(p["temperatura"]),
        "umidade": float(p["umidade"]),
        "pressao": float(p["pressao"]),
        "precipitacao_chuva": precipitacao_chuva,
        "ponto_orvalho": float(p["ponto_orvalho"]),
        "tempo_chuva": tempo_chuva,
        "precipitacao_acumulada": precipitacao_acumulada,
        "solo_elevacao": float(solo_elevacao),

        # REMOVIDO: "chuva_media_h"

        "declive_graus": float(declive_graus),
        "slope_plano": int(slope_plano),
        "declive_bin": declive_bin,

        "intensidade_chuva": intensidade_chuva,
    }
    X = pd.DataFrame([row])

    # 4) Predição
    proba = float(xgb_pipeline.predict_proba(X)[0, 1])
    bucket = bucketize(proba)

    # 5) Resposta
    return PredictResponse(
        risk=round2(proba),
        chance=fmt_ptbr_2dec(proba * 100) + "%",
        bucket=bucket,
        model="xgb",
        meta={
            "declive_graus": round(float(declive_graus), 3),
            "declive_bin": declive_bin,
            "slope_plano": int(slope_plano),
            "solo_elevacao_m": round(float(solo_elevacao), 2),
            "elevacao_google_ok": bool(elev_ok),
            "slope_tif": os.path.basename(SLOPE_TIF),
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
