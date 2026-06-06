import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CATEGORICAL_COLS = [
    'school', 'sex', 'address', 'famsize', 'Pstatus',
    'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
    'famsup', 'paid', 'activities', 'nursery', 'higher',
    'internet', 'romantic'
]

NUMERICAL_COLS = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime',
    'failures', 'famrel', 'freetime', 'goout', 'Dalc',
    'Walc', 'health', 'absences', 'G1', 'G2'
]


def load_uci_data(mat_path=None, por_path=None):
    """Load and optionally merge student-mat and student-por CSVs."""
    if mat_path is None:
        mat_path = os.path.join(BASE_DIR, 'data', 'raw', 'student-mat.csv')
    if por_path is None:
        por_path = os.path.join(BASE_DIR, 'data', 'raw', 'student-por.csv')

    dfs = []
    for path in [mat_path, por_path]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep=';')
                if len(df.columns) <= 1:
                    df = pd.read_csv(path, sep=',')
                dfs.append(df)
            except Exception as e:
                print(f'Warning: Could not load {path}: {e}')

    if not dfs:
        raise FileNotFoundError(
            'No UCI dataset files found. Place student-mat.csv or student-por.csv '
            'in data/raw/ directory. See data/raw/README_DATASETS.txt for download instructions.'
        )

    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f'UCI data loaded: {df.shape[0]} rows, {df.shape[1]} columns')
    return df


def engineer_features(df):
    """Create derived features from existing columns."""
    df = df.copy()
    g1 = df.get('G1', pd.Series([10] * len(df)))
    g2 = df.get('G2', pd.Series([10] * len(df)))
    df['avg_grade'] = (g1 + g2) / 2.0

    dalc = df.get('Dalc', pd.Series([1] * len(df)))
    walc = df.get('Walc', pd.Series([1] * len(df)))
    df['alcohol_risk'] = dalc + walc

    studytime = df.get('studytime', pd.Series([2] * len(df)))
    absences = df.get('absences', pd.Series([0] * len(df)))
    df['study_efficiency'] = studytime / (absences + 1.0)
    return df


def preprocess_uci(df, fit=True, scaler=None, encoder=None, feature_names=None):
    """Full preprocessing: impute, encode, scale, return (X, y, preprocessors, feature_names)."""
    df = engineer_features(df)

    # Target
    if 'G3' in df.columns:
        y = (df['G3'] >= 10).astype(int).values
    else:
        y = None

    engineered = ['avg_grade', 'alcohol_risk', 'study_efficiency']
    all_num = NUMERICAL_COLS + engineered
    num_cols = [c for c in all_num if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    # Impute
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    X_num = num_imputer.fit_transform(df[num_cols])
    X_cat_raw = cat_imputer.fit_transform(df[cat_cols])

    if fit:
        scaler = StandardScaler()
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_num_scaled = scaler.fit_transform(X_num)
        X_cat_enc = encoder.fit_transform(X_cat_raw)
        cat_feature_names = list(encoder.get_feature_names_out(cat_cols))
        feature_names = num_cols + cat_feature_names
    else:
        X_num_scaled = scaler.transform(X_num)
        X_cat_enc = encoder.transform(X_cat_raw)
        cat_feature_names = list(encoder.get_feature_names_out(cat_cols))
        feature_names = num_cols + cat_feature_names

    X = np.hstack([X_num_scaled, X_cat_enc])
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y, scaler, encoder, feature_names


def run_pipeline():
    """End-to-end pipeline: load → engineer → preprocess → save."""
    print('Running UCI preprocessing pipeline...')
    df = load_uci_data()
    X_df, y, scaler, encoder, feature_names = preprocess_uci(df, fit=True)

    # Build output dataframe
    out_df = X_df.copy()
    if y is not None:
        out_df['target'] = y

    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    out_path = os.path.join(processed_dir, 'uci_processed.csv')
    out_df.to_csv(out_path, index=False)
    print(f'Saved processed data to {out_path}')

    scaler_path = os.path.join(models_dir, 'uci_scaler.pkl')
    joblib.dump({'scaler': scaler, 'encoder': encoder, 'feature_names': feature_names}, scaler_path)
    print(f'Saved scaler/encoder to {scaler_path}')
    return out_df, scaler, encoder, feature_names


if __name__ == '__main__':
    run_pipeline()
