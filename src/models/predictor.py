import os
import sys
import numpy as np
import pandas as pd
import joblib
import yaml
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


def load_config():
    cfg_path = os.path.join(BASE_DIR, 'config.yaml')
    with open(cfg_path) as f:
        return yaml.safe_load(f)


FEATURE_DESCRIPTIONS = {
    'failures': 'Past course failures',
    'absences': 'High absenteeism',
    'G1': 'Low first-period grade',
    'G2': 'Low second-period grade',
    'avg_grade': 'Low average grade',
    'studytime': 'Insufficient study time',
    'alcohol_risk': 'High alcohol consumption',
    'study_efficiency': 'Low study efficiency',
    'goout': 'Excessive socializing',
    'total_clicks': 'Low platform engagement',
    'active_days': 'Few active days on platform',
    'avg_score': 'Low assessment scores',
    'submission_rate': 'Missing assignment submissions',
    'late_submission_count': 'Frequent late submissions',
    'early_activity_ratio': 'Weak early engagement',
    'num_of_prev_attempts': 'Multiple previous attempts',
}


class EduPulsePredictor:
    def __init__(self):
        self.cfg = load_config()
        self.models_dir = os.path.join(BASE_DIR, self.cfg['paths']['models_dir'])
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_names = None
        self.dataset = None
        self.model_name = None
        self._bg_cache = {}

    def load_model(self, dataset: str, model_name: str):
        """Load trained model + scaler/encoder for the given dataset."""
        valid_datasets = ['uci', 'oulad']
        valid_models = ['catboost', 'xgboost', 'random_forest', 'mlp', 'svm']
        if dataset not in valid_datasets:
            raise ValueError(f"dataset must be one of {valid_datasets}")
        if model_name not in valid_models:
            raise ValueError(f"model_name must be one of {valid_models}")

        model_path = os.path.join(self.models_dir, f'{dataset}_{model_name}.pkl')
        scaler_path = os.path.join(self.models_dir, f'{dataset}_scaler.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file not found: {model_path}. Run trainer.py first.')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f'Scaler file not found: {scaler_path}. Run preprocessor first.')

        self.model = joblib.load(model_path)
        preprocessor = joblib.load(scaler_path)
        self.scaler = preprocessor['scaler']
        self.encoder = preprocessor['encoder']
        self.feature_names = preprocessor['feature_names']
        self.dataset = dataset
        self.model_name = model_name
        return self

    def _prepare_input(self, input_dict: dict) -> pd.DataFrame:
        """Transform a raw input dict into a scaled feature vector."""
        if self.dataset == 'uci':
            from src.data_pipeline.uci_preprocessor import (
                engineer_features, NUMERICAL_COLS, CATEGORICAL_COLS
            )
            df = pd.DataFrame([input_dict])
            df = engineer_features(df)

            engineered = ['avg_grade', 'alcohol_risk', 'study_efficiency']
            all_num = NUMERICAL_COLS + engineered
            num_cols = [c for c in all_num if c in df.columns]
            cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

            # Fill missing with defaults
            for c in num_cols:
                if c not in df.columns:
                    df[c] = 0.0
            for c in cat_cols:
                if c not in df.columns:
                    df[c] = 'other'

            from sklearn.impute import SimpleImputer
            num_imp = SimpleImputer(strategy='median')
            cat_imp = SimpleImputer(strategy='most_frequent')

            X_num = num_imp.fit_transform(df[num_cols])
            X_cat = cat_imp.fit_transform(df[cat_cols].astype(str))

            X_num_s = self.scaler.transform(X_num)
            X_cat_e = self.encoder.transform(X_cat)

            X = np.hstack([X_num_s, X_cat_e])
        else:  # oulad
            from src.data_pipeline.oulad_preprocessor import CATEGORICAL_COLS, NUMERICAL_COLS
            df = pd.DataFrame([input_dict])

            cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
            num_cols = [c for c in NUMERICAL_COLS if c in df.columns]

            for c in cat_cols:
                if c not in df.columns:
                    df[c] = 'Unknown'
            for c in num_cols:
                if c not in df.columns:
                    df[c] = 0.0

            from sklearn.impute import SimpleImputer
            cat_imp = SimpleImputer(strategy='most_frequent')
            num_imp = SimpleImputer(strategy='median')

            X_cat_raw = cat_imp.fit_transform(df[cat_cols].astype(str))
            X_num_raw = num_imp.fit_transform(df[num_cols])

            X_cat = self.encoder.transform(X_cat_raw)
            X_num = self.scaler.transform(X_num_raw)
            X = np.hstack([X_cat, X_num])

        return pd.DataFrame(X, columns=self.feature_names)

    def _get_risk_factors(self, X_df: pd.DataFrame) -> list:
        """Compute top 3 risk factors via SHAP."""
        model_type = type(self.model).__name__
        try:
            if model_type in ['CatBoostClassifier', 'XGBClassifier', 'RandomForestClassifier']:
                explainer = shap.TreeExplainer(self.model)
                sv = explainer.shap_values(X_df)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) > 1 else sv[0]
                elif len(np.array(sv).shape) == 3:
                    sv = np.array(sv)[:, :, 1]
                sv = np.array(sv).flatten()
            else:
                cache_key = f'{self.dataset}_{self.model_name}'
                if cache_key not in self._bg_cache:
                    processed_path = os.path.join(
                        BASE_DIR, 'data', 'processed', f'{self.dataset}_processed.csv'
                    )
                    if os.path.exists(processed_path):
                        bg_df = pd.read_csv(processed_path).drop(columns=['target'], errors='ignore')
                        self._bg_cache[cache_key] = bg_df.sample(
                            n=min(20, len(bg_df)), random_state=42
                        ).values
                    else:
                        self._bg_cache[cache_key] = X_df.values

                def predict_fn(x):
                    return self.model.predict_proba(x)[:, 1]

                explainer = shap.KernelExplainer(predict_fn, self._bg_cache[cache_key])
                sv = explainer.shap_values(X_df.values, nsamples=15)
                if isinstance(sv, list):
                    sv = sv[0]
                sv = np.array(sv).flatten()

            sorted_idx = np.argsort(sv)  # most negative first (hurting score)
            top3 = []
            for i in sorted_idx[:3]:
                feat = self.feature_names[i]
                desc = FEATURE_DESCRIPTIONS.get(feat.split('_')[0], feat)
                for key, val in FEATURE_DESCRIPTIONS.items():
                    if key in feat.lower():
                        desc = val
                        break
                top3.append(desc)
            return top3
        except Exception as e:
            print(f'SHAP warning: {e}')
            if self.dataset == 'uci':
                return ['Past course failures', 'High absenteeism', 'Low study time']
            return ['Low platform engagement', 'Missing assignments', 'Low assessment scores']

    def predict(self, input_dict: dict) -> dict:
        """Predict student performance from raw input dictionary."""
        if self.model is None:
            raise RuntimeError('Call load_model() first.')

        X_df = self._prepare_input(input_dict)
        pred = int(self.model.predict(X_df)[0])
        proba = self.model.predict_proba(X_df)[0]
        prob_pass = float(proba[1])

        prediction = 'Pass' if pred == 1 else 'Fail'
        confidence = round(proba[pred] * 100, 1)

        if prob_pass >= 0.70:
            risk_level = 'Low'
        elif prob_pass >= 0.50:
            risk_level = 'Medium'
        else:
            risk_level = 'High'

        top_risk_factors = self._get_risk_factors(X_df)

        return {
            'prediction': prediction,
            'confidence': confidence,
            'risk_level': risk_level,
            'top_risk_factors': top_risk_factors,
            'model_used': self.model_name,
            'dataset': self.dataset
        }
