import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import xgboost as xgb

# ==========================
# PARAMÈTRES XGBOOST
# ==========================
xgb_params = {
    "objective": "binary:logistic",  # Classification binaire
    "eval_metric": "logloss",        # Fonction de coût adaptée à la classification
    "nthread": 50,                   # Threads
    "seed": 42,
    "verbosity": 0,
    "learning_rate": 1e-3,
    "max_depth": 5,
    "tree_method": "hist",           # Plus rapide pour gros datasets
}

NUM_BOOST_ROUND = 100

# ==========================
# GÉNÉRATION DES DATES DE TRAIN/TEST
# ==========================

path = 'data/'

X_train = pd.read_csv(path + 'X_train.csv',index_col='ROW_ID')
# X_test = pd.read_csv(path + 'X_test.csv',index_col='ROW_ID')

y_train = pd.read_csv(path + 'y_train.csv',index_col='ROW_ID')
# sample_submission = pd.read_csv(path + 'sample_submission.csv',index_col='ROW_ID')

RET_features = [f'RET_{i}' for i in range(1,20)]
SIGNED_VOLUME_features = [f'SIGNED_VOLUME_{i}' for i in range(1,20)]
TURNOVER_features = ['AVG_DAILY_TURNOVER']
for i in [3,5,10,15,20]:
    X_train[ f'AVERAGE_PERF_{i}'] = X_train[RET_features[:i]].mean(1)
    X_train[ f'ALLOCATIONS_AVERAGE_PERF_{i}'] = X_train.groupby('TS')[ f'AVERAGE_PERF_{i}'].transform('mean')
    
# features = RET_features + SIGNED_VOLUME_features + TURNOVER_features
features = TURNOVER_features
features = features + [ f'AVERAGE_PERF_{i}' for i in [3,5,10,15,20]]
features = features + [ f'ALLOCATIONS_AVERAGE_PERF_{i}' for i in [3,5,10,15,20]]




# ==========================
# BOUCLE DE VALIDATION CROISÉE
# ==========================
for allocation in X_train['ALLOCATION'].unique()[:10]:
    X_train_alloc = X_train[X_train['ALLOCATION'] == allocation]
    y_train_alloc = y_train.loc[X_train_alloc.index]


    train_dates = X_train_alloc['TS'].unique()
    # test_dates = X_test['TS'].unique()

    n_splits = 10
    scores_xgb = []
    models_xgb = []

    splits = KFold(
        n_splits=n_splits,
        random_state=0,
        shuffle=True
    ).split(train_dates)

    print(f"\n=== ALLOCATION {allocation} ===")
    for i, (local_train_dates_ids, local_test_dates_ids) in enumerate(splits):
        local_train_dates = train_dates[local_train_dates_ids]
        local_test_dates = train_dates[local_test_dates_ids]

        local_train_ids = X_train_alloc['TS'].isin(local_train_dates)
        local_test_ids = X_train_alloc['TS'].isin(local_test_dates)

        X_local_train = X_train_alloc.loc[local_train_ids, features]
        y_local_train = y_train_alloc.loc[local_train_ids, 'target']
        X_local_test = X_train_alloc.loc[local_test_ids, features]
        y_local_test = y_train_alloc.loc[local_test_ids, 'target']

        # Transformation binaire : y > 0
        y_local_train_bin = (y_local_train > 0).astype(int)
        y_local_test_bin = (y_local_test > 0).astype(int)

        # Conversion en DMatrix (format natif XGBoost)
        dtrain = xgb.DMatrix(X_local_train, label=y_local_train_bin)
        dtest = xgb.DMatrix(X_local_test, label=y_local_test_bin)

        # Entraînement
        model_xgb = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=NUM_BOOST_ROUND,
        )

        # Prédiction
        y_local_pred_proba = model_xgb.predict(dtest)
        y_local_pred = (y_local_pred_proba > 0.5).astype(int)

        # Score
        score = accuracy_score(y_local_test_bin, y_local_pred)
        scores_xgb.append(score)
        models_xgb.append(model_xgb)

        print(f"Fold {i+1} - Accuracy: {score * 100:.2f}%")

    # ==========================
    # RÉSUMÉ DES SCORES
    # ==========================
    mean = np.mean(scores_xgb) * 100
    std = np.std(scores_xgb) * 100
    u = (mean + std)
    l = (mean - std)

    print(f'Accuracy: {mean:.2f}% [{l:.2f} ; {u:.2f}] (+- {std:.2f})')
