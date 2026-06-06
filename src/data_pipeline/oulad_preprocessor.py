import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CATEGORICAL_COLS = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
NUMERICAL_COLS = [
    'num_of_prev_attempts', 'studied_credits',
    'total_clicks', 'avg_clicks_per_day', 'active_days', 'last_active_day',
    'early_activity_ratio', 'avg_score', 'submission_rate',
    'late_submission_count', 'weighted_avg_score'
]


def load_oulad_data(info_path=None, vle_path=None, assess_path=None, meta_path=None):
    raw_dir = os.path.join(BASE_DIR, 'data', 'raw')
    if info_path is None:
        info_path = os.path.join(raw_dir, 'studentInfo.csv')
    if vle_path is None:
        vle_path = os.path.join(raw_dir, 'studentVle.csv')
    if assess_path is None:
        assess_path = os.path.join(raw_dir, 'studentAssessment.csv')
    if meta_path is None:
        meta_path = os.path.join(raw_dir, 'assessments.csv')

    for p in [info_path, vle_path, assess_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'OULAD file not found: {p}\n'
                'See data/raw/README_DATASETS.txt for download instructions.'
            )

    info_df = pd.read_csv(info_path)
    vle_df = pd.read_csv(vle_path)
    assess_df = pd.read_csv(assess_path)
    meta_df = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    print(f'OULAD loaded: {len(info_df)} students, {len(vle_df)} VLE rows, {len(assess_df)} assessment rows')
    return info_df, vle_df, assess_df, meta_df


def engineer_vle_features(vle_df):
    grp = vle_df.groupby('id_student')
    total_clicks = grp['sum_click'].sum().rename('total_clicks')
    active_days = vle_df.groupby('id_student')['date'].nunique().rename('active_days')
    last_active_day = grp['date'].max().rename('last_active_day')
    early_clicks = vle_df[vle_df['date'] <= 30].groupby('id_student')['sum_click'].sum()
    early_ratio = (early_clicks / total_clicks).rename('early_activity_ratio').fillna(0.0)
    avg_clicks = (total_clicks / active_days).rename('avg_clicks_per_day').fillna(0.0)
    return pd.concat([total_clicks, active_days, last_active_day, early_ratio, avg_clicks], axis=1)


def engineer_assessment_features(assess_df, meta_df):
    if meta_df is not None:
        merged = assess_df.merge(meta_df[['id_assessment', 'date', 'weight']], on='id_assessment', how='left')
    else:
        merged = assess_df.copy()
        merged['date'] = 0
        merged['weight'] = 1

    merged['score'] = pd.to_numeric(merged['score'], errors='coerce').fillna(0)
    merged['weight'] = pd.to_numeric(merged.get('weight', 1), errors='coerce').fillna(1)
    merged['date'] = pd.to_numeric(merged.get('date', 0), errors='coerce').fillna(0)
    merged['date_submitted'] = pd.to_numeric(merged.get('date_submitted', 0), errors='coerce').fillna(0)

    grp = merged.groupby('id_student')
    avg_score = grp['score'].mean().rename('avg_score')
    submission_count = grp['id_assessment'].count().rename('submission_count')
    merged['is_late'] = (merged['date_submitted'] > merged['date']).astype(int)
    late_count = grp['is_late'].sum().rename('late_submission_count')
    merged['weighted_score'] = merged['score'] * merged['weight']
    weight_sum = grp['weight'].sum()
    weighted_avg = (grp['weighted_score'].sum() / weight_sum).rename('weighted_avg_score').fillna(0)

    total_assessments = len(meta_df) if meta_df is not None else grp['id_assessment'].nunique().max()
    total_assessments = max(total_assessments, 1)
    submission_rate = (submission_count / total_assessments).clip(0, 1).rename('submission_rate')

    return pd.concat([avg_score, submission_rate, late_count, weighted_avg], axis=1)


def merge_oulad(info_df, vle_df, assess_df, meta_df):
    vle_features = engineer_vle_features(vle_df)
    ass_features = engineer_assessment_features(assess_df, meta_df)
    df = info_df.merge(vle_features, on='id_student', how='left')
    df = df.merge(ass_features, on='id_student', how='left')
    fill_cols = ['total_clicks', 'active_days', 'last_active_day', 'early_activity_ratio',
                 'avg_clicks_per_day', 'avg_score', 'submission_rate', 'late_submission_count', 'weighted_avg_score']
    df[fill_cols] = df[fill_cols].fillna(0.0)
    df['target'] = df['final_result'].apply(lambda x: 1 if x in ['Pass', 'Distinction'] else 0)
    return df


def preprocess_oulad(df, fit=True, scaler=None, encoder=None, feature_names=None):
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]

    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = SimpleImputer(strategy='median')

    X_cat_raw = cat_imputer.fit_transform(df[cat_cols].astype(str))
    X_num_raw = num_imputer.fit_transform(df[num_cols])

    if fit:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        scaler = StandardScaler()
        X_cat = encoder.fit_transform(X_cat_raw)
        X_num = scaler.fit_transform(X_num_raw)
        feature_names = cat_cols + num_cols
    else:
        X_cat = encoder.transform(X_cat_raw)
        X_num = scaler.transform(X_num_raw)
        feature_names = cat_cols + num_cols

    X = np.hstack([X_cat, X_num])
    X_df = pd.DataFrame(X, columns=feature_names)
    y = df['target'].values if 'target' in df.columns else None
    return X_df, y, scaler, encoder, feature_names


def run_pipeline():
    print('Running OULAD preprocessing pipeline...')
    info_df, vle_df, assess_df, meta_df = load_oulad_data()
    df = merge_oulad(info_df, vle_df, assess_df, meta_df)
    X_df, y, scaler, encoder, feature_names = preprocess_oulad(df, fit=True)

    out_df = X_df.copy()
    if y is not None:
        out_df['target'] = y

    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    out_path = os.path.join(processed_dir, 'oulad_processed.csv')
    out_df.to_csv(out_path, index=False)
    print(f'Saved processed data to {out_path}')

    scaler_path = os.path.join(models_dir, 'oulad_scaler.pkl')
    joblib.dump({'scaler': scaler, 'encoder': encoder, 'feature_names': feature_names}, scaler_path)
    print(f'Saved scaler/encoder to {scaler_path}')
    return out_df, scaler, encoder, feature_names


if __name__ == '__main__':
    run_pipeline()
