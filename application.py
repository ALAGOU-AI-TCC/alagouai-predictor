from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import os
from joblib import load as joblib_load
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from decimal import Decimal, ROUND_HALF_UP

# (opcional) esconder o warning de versão, enquanto você não alinha o scikit-learn/xgboost
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

MODELS_DIR = os.getenv("MODELS_DIR", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "modelo_xgboost.pkl")  # seu arquivo .pkl

app = FastAPI(title="Flood Risk Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- carrega e garante que é um modelo válido ---
xgb_pipeline = joblib_load(MODEL_PATH)
if not hasattr(xgb_pipeline, "predict_proba"):
    raise RuntimeError(f"{MODEL_PATH} não parece um Pipeline com predict_proba.")

# Nomes EXATOS usados no treino (ajuste se necessário)
MODEL_FEATURES = [
    "temperatura", "umidade", "pressao",
    "precipitacao_chuva", "ponto_orvalho",
    "tempo_chuva", "precipitacao_acumulada",
    "intensidade_chuva"
]

# Input aceita camelCase no JSON e mapeia para snake_case internamente
class Input(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    temperatura: float = Field(..., alias="temperatura")
    umidade: float = Field(..., alias="umidade")
    pressao: float = Field(..., alias="pressao")

    precipitacao_chuva: float = Field(..., alias="precipitacaoChuva")
    ponto_orvalho: float = Field(..., alias="pontoOrvalho")
    tempo_chuva: int = Field(..., alias="tempoChuva")
    precipitacao_acumulada: float = Field(..., alias="precipitacaoAcumulada")
    intensidade_chuva: str = Field(..., alias="intensidadeChuva")  # OHE no pipeline

class PredictResponse(BaseModel):
    risk: float      # 0..1 (duas casas)
    chance: str      # "pt-BR" com duas casas e '%', ex.: "40,20%"
    bucket: str
    model: str

def bucketize(p: float) -> str:
    if p <= 0.30: return "baixo"
    if p <= 0.70: return "medio"
    return "alto"

def round2(v: float) -> float:
    # arredondamento estável half-up
    return float(Decimal(v).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def fmt_ptbr_2dec(v: float) -> str:
    # retorna string com vírgula e 2 casas, ex.: "40,20"
    s = f"{round2(v):.2f}"
    return s.replace(".", ",")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(inp: Input):
    # 1) Pega dados internos (snake_case)
    payload = inp.model_dump(by_alias=False)

    # 2) DataFrame com nomes internos
    df_internal = pd.DataFrame([payload])

    # 3) Valida features exigidas pelo modelo
    missing = [c for c in MODEL_FEATURES if c not in df_internal.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"error": "Campos ausentes para o modelo", "missing_features": missing}
        )

    # 4) Ordena as colunas na ordem do treino
    X = df_internal.reindex(columns=MODEL_FEATURES)

    # 5) Predição
    proba = float(xgb_pipeline.predict_proba(X)[0, 1])  # 0..1
    bucket = bucketize(proba)

    # 6) Formatação: risk (numérico 0..1 com 2 casas) e chance (string pt-BR em %)
    risk_num = round2(proba)                 # e.g., 0.40
    chance_pct_str = fmt_ptbr_2dec(proba * 100) + "%"   # e.g., "40,20%"

    return PredictResponse(
        risk=risk_num,
        chance=chance_pct_str,
        bucket=bucket,
        model="xgb"
    )

if __name__ == "__main__":
    import uvicorn
    # Rodar clicando no arquivo (sem reload)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
