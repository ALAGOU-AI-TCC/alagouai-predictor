# model_bundle.py
import pandas as pd

class PreprocessXGB:
    def __init__(self, preprocessor, xgb, feature_order):
        self.preprocessor = preprocessor
        self.xgb = xgb
        self.feature_order = feature_order

    def _to_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature_order]
        return pd.DataFrame(X, columns=self.feature_order)

    def predict_proba(self, X):
        Xdf = self._to_df(X)
        Xt = self.preprocessor.transform(Xdf)
        return self.xgb.predict_proba(Xt)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
