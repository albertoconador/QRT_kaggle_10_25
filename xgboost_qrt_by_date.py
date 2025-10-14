import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import xgboost as xgb
from matplotlib import pyplot as plt
from tqdm import tqdm
import optuna

# ==========================
# PARAMÈTRES XGBOOST
# ==========================
# xgb_params = {
#     "objective": "binary:logistic",
#     "eval_metric": "logloss",        # ou "auc" selon ton critère
#     "booster": "gbtree",
#     # "n_estimators": 500,             # nombre d’arbres (ajuste avec early_stopping)
#     "learning_rate": 0.03,           # petit taux pour stabilité temporelle
#     "max_depth": 4,                  # profondeur limitée (évite overfit)
#     "min_child_weight": 3,           # contrôle la complexité
#     "subsample": 0.7,                # sous-échantillonnage des données
#     "colsample_bytree": 0.7,         # sous-échantillonnage des features
#     "gamma": 0.1,                    # régularisation supplémentaire
#     "lambda": 1.0,                   # L2 regularization
#     "alpha": 0.5,                    # L1 regularization
#     "scale_pos_weight": 1,           # ajuste si classes déséquilibrées
#     "random_state": 42,
#     "tree_method": "hist",           # rapide et efficace
# }

plot_scores_xgb = {}

NUM_BOOST_ROUND = 500

# ==========================
# GÉNÉRATION DES DATES DE TRAIN/TEST
# ==========================

path = 'data/'

X_train = pd.read_csv(path + 'X_train.csv',index_col='ROW_ID')
# X_test = pd.read_csv(path + 'X_test.csv',index_col='ROW_ID')

y_train = pd.read_csv(path + 'y_train.csv',index_col='ROW_ID')
# sample_submission = pd.read_csv(path + 'sample_submission.csv',index_col='ROW_ID')

def FAR(row):
    ret = np.array(row['ret_ts'])
    vol = np.array(row['vol_ts'])
    return np.sum(ret * vol) / np.sum(np.abs(vol))



RET_features = [f'RET_{i}' for i in range(1,20)]
SIGNED_VOLUME_features = [f'SIGNED_VOLUME_{i}' for i in range(1,20)]
TURNOVER_features = ['AVG_DAILY_TURNOVER']
for i in [3,5,10,15,20]:
    X_train[ f'AVERAGE_PERF_{i}'] = X_train[RET_features[:i]].mean(1)
    X_train[ f'ALLOCATIONS_AVERAGE_PERF_{i}'] = X_train.groupby('TS')[ f'AVERAGE_PERF_{i}'].transform('mean')
    
X_train['ret_ts'] = X_train[RET_features].values.tolist()
X_train['vol_ts'] = X_train[SIGNED_VOLUME_features].values.tolist()
X_train['FAR'] = X_train.apply(FAR, axis=1)
features = RET_features + SIGNED_VOLUME_features + TURNOVER_features + ['FAR']
features = features + [ f'AVERAGE_PERF_{i}' for i in [3,5,10,15,20]]
features = features + [ f'ALLOCATIONS_AVERAGE_PERF_{i}' for i in [3,5,10,15,20]]




# ==========================
# BOUCLE DE VALIDATION CROISÉE
# ==========================
for date in tqdm(X_train['TS'].unique()[:]):
    X_train_date = X_train[X_train['TS'] == date]
    y_train_date = y_train.loc[X_train_date.index]


    train_alloc = X_train_date['ALLOCATION'].unique()
    # test_dates = X_test['TS'].unique()

    n_splits = 10
    scores_xgb = []
    models_xgb = []

    splits = KFold(
        n_splits=n_splits,
        random_state=0,
        shuffle=True
    ).split(train_alloc)

    def objective(trial):
        # === Hyperparamètres à explorer ===
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "booster": "gbtree",
            "eta": trial.suggest_float("learning_rate", 0.01, 0.2, log=False),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "lambda": trial.suggest_float("lambda", 0.5, 3.0, log=False),
            "alpha": trial.suggest_float("alpha", 0.0, 1.0),
        }

        # print(f"\n=== ALLOCATION {allocation} ===")
        for i, (local_train_alloc_ids, local_test_alloc_ids) in enumerate(splits):
            local_train_dates = train_alloc[local_train_alloc_ids]
            local_test_dates = train_alloc[local_test_alloc_ids]

            local_train_ids = X_train_date['ALLOCATION'].isin(local_train_dates)
            local_test_ids = X_train_date['ALLOCATION'].isin(local_test_dates)

            X_local_train = X_train_date.loc[local_train_ids, features]
            y_local_train = y_train_date.loc[local_train_ids, 'target']
            X_local_test = X_train_date.loc[local_test_ids, features]
            y_local_test = y_train_date.loc[local_test_ids, 'target']

            # Transformation binaire : y > 0
            y_local_train_bin = (y_local_train > 0).astype(int)
            y_local_test_bin = (y_local_test > 0).astype(int)

            # Conversion en DMatrix (format natif XGBoost)
            dtrain = xgb.DMatrix(X_local_train, label=y_local_train_bin)
            dtest = xgb.DMatrix(X_local_test, label=y_local_test_bin)

            # Entraînement
            model_xgb = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=NUM_BOOST_ROUND,
                evals=[(dtrain, "train"), (dtest, "test")],
                early_stopping_rounds=30,
                verbose_eval=False
            )

            # Prédiction
            y_local_pred_proba = model_xgb.predict(dtest)
            y_local_pred = (y_local_pred_proba > 0.5).astype(int)

            # Score
            score = accuracy_score(y_local_test_bin, y_local_pred)
            scores_xgb.append(score)
            models_xgb.append(model_xgb)

        # print(f"Fold {i+1} - Accuracy: {score * 100:.2f}%")

        # ==========================
        # RÉSUMÉ DES SCORES
        # ==========================
        mean = np.mean(scores_xgb) * 100
        std = np.std(scores_xgb) * 100
        u = (mean + std)
        l = (mean - std)

        # print(f'Accuracy: {mean:.2f}% [{l:.2f} ; {u:.2f}] (+- {std:.2f})')
        trial.set_user_attr("std", std)
        return mean


    # === Lancement de l’optimisation bayésienne ===
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    # Retrieve best results for this date
    best_mean = study.best_value
    best_params = study.best_params
    best_std = study.best_trial.user_attrs.get("std", None)

    print("Best params:", best_params)
    print("Best mean accuracy (%):", best_mean)
    if best_std is not None:
        print("Best std (%):", best_std)

    # Store for plotting
    plot_scores_xgb[date] = (best_mean, best_std if best_std is not None else 0.0)


# ==========================
# plot scores xgboost
# ==========================

dates = list(plot_scores_xgb.keys())
means = [plot_scores_xgb[date][0] for date in dates]
stds = [plot_scores_xgb[date][1] for date in dates]

plt.figure(figsize=(10, 6))
plt.bar(dates, means, yerr=stds, capsize=5)
plt.hlines(y=50, xmin=-1, xmax=len(dates), colors='r', linestyles='dashed', label='Baseline 50%')
plt.hlines(y=53.722, xmin=-1, xmax=len(dates), colors='k', linestyles='dashed', label='Baseline 53.722%')
plt.xlabel('Date')
plt.ylabel('Accuracy (%)')
plt.title('XGBoost Accuracy by Date with Standard Deviation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

