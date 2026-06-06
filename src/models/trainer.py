import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import joblib
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.utils.metrics import evaluate_model, get_shap_values


def load_config():
    cfg_path = os.path.join(BASE_DIR, 'config.yaml')
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def build_models(cfg, random_state=42):
    mc = cfg['models']
    return {
        'catboost': CatBoostClassifier(
            iterations=mc['catboost']['iterations'],
            learning_rate=mc['catboost']['learning_rate'],
            depth=mc['catboost']['depth'],
            verbose=0,
            random_seed=random_state
        ),
        'xgboost': XGBClassifier(
            n_estimators=mc['xgboost']['n_estimators'],
            max_depth=mc['xgboost']['max_depth'],
            learning_rate=mc['xgboost']['learning_rate'],
            eval_metric='logloss',
            random_state=random_state
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=mc['random_forest']['n_estimators'],
            max_depth=mc['random_forest']['max_depth'],
            min_samples_split=mc['random_forest']['min_samples_split'],
            random_state=random_state
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes=tuple(mc['mlp']['hidden_layer_sizes']),
            max_iter=mc['mlp']['max_iter'],
            early_stopping=mc['mlp']['early_stopping'],
            random_state=random_state
        ),
        'svm': SVC(
            kernel=mc['svm']['kernel'],
            C=mc['svm']['C'],
            probability=mc['svm']['probability'],
            random_state=random_state
        )
    }


def train_on_dataset(dataset_name, cfg):
    processed_dir = os.path.join(BASE_DIR, cfg['paths']['processed_dir'])
    models_dir = os.path.join(BASE_DIR, cfg['paths']['models_dir'])
    os.makedirs(models_dir, exist_ok=True)

    csv_path = os.path.join(processed_dir, f'{dataset_name}_processed.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f'Processed CSV not found: {csv_path}\n'
            f'Run: python src/data_pipeline/{dataset_name}_preprocessor.py first.'
        )

    df = pd.read_csv(csv_path)
    if 'target' not in df.columns:
        raise ValueError(f"'target' column missing from {csv_path}")

    X = df.drop(columns=['target']).values
    y = df['target'].values
    feature_names = list(df.drop(columns=['target']).columns)

    rs = cfg['training']['random_state']
    ts = cfg['training']['test_size']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ts, stratify=y, random_state=rs
    )

    if cfg['training']['use_smote']:
        print(f'  Applying SMOTE: {np.bincount(y_train)} -> ', end='')
        smote = SMOTE(random_state=rs)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(np.bincount(y_train))

    models = build_models(cfg, rs)
    cv = StratifiedKFold(n_splits=cfg['training']['cv_folds'], shuffle=True, random_state=rs)

    # GridSearchCV for CatBoost and XGBoost
    cb_params = {'depth': [4, 6], 'learning_rate': [0.05, 0.1]}
    cb_grid = GridSearchCV(models['catboost'], cb_params, cv=cv, scoring='accuracy', n_jobs=-1)

    xgb_params = {'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
    xgb_grid = GridSearchCV(models['xgboost'], xgb_params, cv=cv, scoring='accuracy', n_jobs=-1)

    results = {}
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    for name, model in models.items():
        print(f'  Training {name}...')
        if name == 'catboost':
            cb_grid.fit(X_train, y_train)
            model = cb_grid.best_estimator_
            print(f'    Best params: {cb_grid.best_params_}')
        elif name == 'xgboost':
            xgb_grid.fit(X_train, y_train)
            model = xgb_grid.best_estimator_
            print(f'    Best params: {xgb_grid.best_params_}')
        else:
            model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test_df, y_test, feature_names)
        top3 = get_shap_values(model, X_test_df, feature_names)
        metrics['top_features'] = top3
        results[name] = metrics

        model_path = os.path.join(models_dir, f'{dataset_name}_{name}.pkl')
        joblib.dump(model, model_path)
        print(f'    Saved → {model_path} | Acc: {metrics["accuracy"]:.4f} | F1: {metrics["f1_score"]:.4f} | AUC: {metrics["roc_auc"]:.4f}')

    return results


def print_summary_table(all_results):
    print('\n' + '=' * 80)
    print('  MODEL PERFORMANCE SUMMARY')
    print('=' * 80)
    header = f'{"Model":<20} {"Dataset":<8} {"Accuracy":>10} {"F1-Score":>10} {"ROC-AUC":>10}'
    print(header)
    print('-' * 60)
    for dataset, results in all_results.items():
        for model_name, metrics in results.items():
            print(f'{model_name:<20} {dataset:<8} {metrics["accuracy"]:>10.4f} {metrics["f1_score"]:>10.4f} {metrics["roc_auc"]:>10.4f}')
    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(description='Train EduPulse ML models')
    parser.add_argument('--dataset', choices=['uci', 'oulad', 'both'], default='both')
    args = parser.parse_args()

    cfg = load_config()
    models_dir = os.path.join(BASE_DIR, cfg['paths']['models_dir'])
    comparison_path = os.path.join(models_dir, 'model_comparison.json')

    # Load existing comparison if available
    all_results = {}
    if os.path.exists(comparison_path):
        with open(comparison_path) as f:
            all_results = json.load(f)

    datasets = ['uci', 'oulad'] if args.dataset == 'both' else [args.dataset]

    for ds in datasets:
        print(f'\n{"="*50}')
        print(f'  Training on {ds.upper()} dataset')
        print(f'{"="*50}')
        try:
            all_results[ds] = train_on_dataset(ds, cfg)
        except FileNotFoundError as e:
            print(f'Skipping {ds}: {e}')

    with open(comparison_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved comparison → {comparison_path}')

    print_summary_table(all_results)


if __name__ == '__main__':
    main()
