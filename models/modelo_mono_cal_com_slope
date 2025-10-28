# treino_xgb_mono_cal_bins.py
# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from pathlib import Path
import joblib, json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.isotonic import IsotonicRegression

import xgboost as xgb

# =======================
# 0) Carrega dados
# =======================
df = pd.read_csv('../data/processed/dados_processados.csv')

# Clippings e tipos
df['tempo_chuva'] = df['tempo_chuva'].clip(0, 4).astype('int8')
df['precipitacao_chuva'] = df['precipitacao_chuva'].clip(lower=0)
df['precipitacao_acumulada'] = df['precipitacao_acumulada'].clip(lower=0)
df['declive_graus'] = pd.to_numeric(df['declive_graus'], errors='coerce').clip(lower=0)

# Engenharia de terreno
declive_bins = [0,2,4,6,8,10,15,60]
declive_labels = [f'{declive_bins[i]}–{declive_bins[i+1]}°' for i in range(len(declive_bins)-1)]
df['declive_bin'] = pd.cut(df['declive_graus'], bins=declive_bins, labels=declive_labels,
                           include_lowest=True, right=False)
df['slope_plano'] = (df['declive_graus'] < 2).astype('int8')

# Remove colunas não-preditoras
drop_cols = ["historico_alagamento", "data_hora", "latitude",
             "longitude", "bairro", "precipitacao_diaria", "velocidade_vento"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
y = df["historico_alagamento"].astype(int)

# Garante que não entrou nada que não queremos (ex.: chuva_media_h)
if 'chuva_media_h' in X.columns:
    X = X.drop(columns=['chuva_media_h'])

# Listas de features
numerical_features = [
    'temperatura', 'umidade', 'pressao',
    'precipitacao_chuva', 'ponto_orvalho',
    'precipitacao_acumulada', 'solo_elevacao',
    'tempo_chuva',
    'declive_graus',
    'slope_plano'
]
categorical_features = [c for c in ['intensidade_chuva', 'declive_bin'] if c in X.columns]

# =======================
# 1) Split (treino/val)
# =======================
Xtr, Xcal, y_tr, y_cal = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# =======================
# 2) Preprocessador
# =======================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', [c for c in numerical_features if c in X.columns]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# Ajusta só no treino
preprocessor.fit(Xtr)

# Transforma
Xt_tr  = preprocessor.transform(Xtr)
Xt_cal = preprocessor.transform(Xcal)

# =======================
# 3) Treino XGBoost (xgb.train)
# =======================
# Lida com desbalanceamento
pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
scale_pos_weight = neg / max(pos, 1)

dtr  = xgb.DMatrix(Xt_tr,  label=y_tr)
dcal = xgb.DMatrix(Xt_cal, label=y_cal)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "lambda": 1.0,
    "gamma": 0.0,
    "seed": 42,
    "tree_method": "hist",
    "scale_pos_weight": scale_pos_weight,
}
watchlist = [(dtr, "train"), (dcal, "valid")]
booster = xgb.train(
    params=params,
    dtrain=dtr,
    num_boost_round=1200,
    evals=watchlist,
    early_stopping_rounds=80,
    verbose_eval=False
)

# =======================
# 4) Calibração condicional (isotônica)
# =======================
def no_rain_mask(df_):
    pc = df_.get('precipitacao_chuva', 0).fillna(0).astype(float)
    pa = df_.get('precipitacao_acumulada', 0).fillna(0).astype(float)
    tc = df_.get('tempo_chuva', 0).fillna(0).astype(float)
    return ((pc == 0.0) & (pa == 0.0) & (tc == 0.0)).to_numpy()

p_raw_cal = booster.predict(dcal, iteration_range=(0, booster.best_iteration + 1))
nr_mask = no_rain_mask(Xcal)

# modelos isotônicos
def fit_iso(x, y):
    # protege contra casos degenerados
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 10 or len(np.unique(x)) < 3:
        # volta identidade (sem calibração) se não der para ajustar
        iso = IsotonicRegression(out_of_bounds="clip")
        # ajusta com algo mínimo
        xx = np.linspace(0,1,5)
        iso.fit(xx, xx)
        return iso
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y)
    return iso

iso_all    = fit_iso(p_raw_cal, y_cal)
iso_norain = fit_iso(p_raw_cal[nr_mask], y_cal[nr_mask]) if nr_mask.any() else None
iso_rain   = fit_iso(p_raw_cal[~nr_mask], y_cal[~nr_mask]) if (~nr_mask).any() else None

# =======================
# 5) Wrapper para servir
# =======================
class MonoCalibratedModel:
    """
    Wrapper: preprocessor -> booster (xgb.train) -> calibração isotônica (global/condicional).
    Exponde predict_proba(X_df) que retorna n×2 (sklearn-like).
    """
    def __init__(self, preprocessor, booster, iso_all, iso_norain, iso_rain,
                 num_features, cat_features):
        self.preprocessor = preprocessor
        self.booster = booster
        self.iso_all = iso_all
        self.iso_norain = iso_norain
        self.iso_rain = iso_rain
        self.num_features = num_features
        self.cat_features = cat_features

    def _no_rain_mask(self, X_df: pd.DataFrame):
        pc = X_df.get('precipitacao_chuva', 0).fillna(0).astype(float)
        pa = X_df.get('precipitacao_acumulada', 0).fillna(0).astype(float)
        tc = X_df.get('tempo_chuva', 0).fillna(0).astype(float)
        return ((pc == 0.0) & (pa == 0.0) & (tc == 0.0)).to_numpy()

    def predict_proba(self, X_df: pd.DataFrame):
        Xt = self.preprocessor.transform(X_df[self.num_features + self.cat_features])
        dX = xgb.DMatrix(Xt)
        try:
            p_raw = self.booster.predict(dX, iteration_range=(0, self.booster.best_iteration + 1))
        except Exception:
            p_raw = self.booster.predict(dX)

        nr_mask = self._no_rain_mask(X_df)
        p_cal = np.empty_like(p_raw, dtype=float)

        # sem chuva
        if (nr_mask == 1).any():
            p_slice = p_raw[nr_mask]
            if self.iso_norain is not None:
                p_cal[nr_mask] = self.iso_norain.transform(p_slice) if p_slice.size > 0 else p_slice
            else:
                p_cal[nr_mask] = self.iso_all.transform(p_slice) if p_slice.size > 0 else p_slice

        # com chuva
        rain_mask = ~nr_mask
        if rain_mask.any():
            p_slice = p_raw[rain_mask]
            if self.iso_rain is not None:
                p_cal[rain_mask] = self.iso_rain.transform(p_slice) if p_slice.size > 0 else p_slice
            else:
                p_cal[rain_mask] = self.iso_all.transform(p_slice) if p_slice.size > 0 else p_slice

        p_cal = np.clip(p_cal, 0.0, 1.0)
        return np.vstack([1.0 - p_cal, p_cal]).T

    def predict(self, X_df: pd.DataFrame):
        return (self.predict_proba(X_df)[:, 1] >= 0.50).astype(int)

# =======================
# 6) Métricas simples e salvamento
# =======================
# holdout report (com corte 0.5 – apenas informativo)
y_cal_pred = (booster.predict(dcal) >= 0.5).astype(int)
print("\nRelatório (holdout, pre-calibração):")
print(classification_report(y_cal, y_cal_pred, digits=3))
print("ROC AUC (holdout, pre-calibração):", roc_auc_score(y_cal, booster.predict(dcal)))

# salva artefatos
Path('../models').mkdir(parents=True, exist_ok=True)
Path('../reports').mkdir(parents=True, exist_ok=True)

serving = MonoCalibratedModel(
    preprocessor=preprocessor,
    booster=booster,
    iso_all=iso_all,
    iso_norain=iso_norain,
    iso_rain=iso_rain,
    num_features=[c for c in numerical_features if c in X.columns],
    cat_features=categorical_features
)
joblib.dump(serving, '../models/modelo_xgb_mono_cal_comslope.pkl')

with open('../reports/declive_bins.json', 'w', encoding='utf-8') as f:
    json.dump({"bins": declive_bins, "labels": declive_labels}, f, ensure_ascii=False, indent=2)

print("\nOK - Modelo salvo em ../models/modelo_xgb_mono_cal.pkl")
