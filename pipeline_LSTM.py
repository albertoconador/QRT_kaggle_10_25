import pandas as pd
import numpy as np
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Dropout, TimeDistributed
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from itertools import product
import tensorflow as tf

from keras.callbacks import TensorBoard
import os

# Crée un dossier pour les logs
log_dir = os.path.join("logs", "fit")
os.makedirs(log_dir, exist_ok=True)

# Crée un dossier pour les modèles
os.makedirs('models', exist_ok=True)


# Crée le callback TensorBoard
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=0
)




def import_and_prepoccess_data(path = 'data/'):
    path = 'data/'

    X_train = pd.read_csv(path + 'X_train.csv',index_col='ROW_ID')
    X_test = pd.read_csv(path + 'X_test.csv',index_col='ROW_ID')

    y_train = pd.read_csv(path + 'y_train.csv',index_col='ROW_ID')
    y_train = (y_train > 0).astype(int)
    sample_submission = pd.read_csv(path + 'sample_submission.csv',index_col='ROW_ID')

    RET_features = [f'RET_{i}' for i in range(1,21)]
    SIGNED_VOLUME_features = [f'SIGNED_VOLUME_{i}' for i in range(1,21)]
    TURNOVER_features = ['AVG_DAILY_TURNOVER']

    RET_features_subset = []
    for i in [3,5,10,15,20]:
        RET_features_subset.append(f'RET_{i}')

    def FAR(row):
        ret = np.array(row['ret_ts'])
        vol = np.array(row['vol_ts'])
        return np.sum(ret * vol) / np.sum(np.abs(vol))
    
    X_train['ret_ts'] = X_train[RET_features].values.tolist()
    X_train['vol_ts'] = X_train[SIGNED_VOLUME_features].values.tolist()
    X_train['FAR'] = X_train.apply(FAR, axis=1)
    X_test['ret_ts'] = X_test[RET_features].values.tolist()
    X_test['vol_ts'] = X_test[SIGNED_VOLUME_features].values.tolist()
    X_test['FAR'] = X_test.apply(FAR, axis=1)
    

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'sample_submission': sample_submission,
        'RET_features': RET_features,
        'SIGNED_VOLUME_features': SIGNED_VOLUME_features,
        'TURNOVER_features': TURNOVER_features,
        'all_features': RET_features + SIGNED_VOLUME_features + TURNOVER_features + ['FAR'],
    }

def build_lstm_model(input_shape, lstm_units=50):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(Dropout(param_grid['dropout']))
    model.add(TimeDistributed(Dense(units=1, activation='sigmoid')))  # Binary classification
    model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model

param_grid = {
        'units': 50,
        'epochs': 20,
        'batch_size': 16,
        'dropout': 0.2
    }

def optimize(data, allocations, n_splits=4):

    dates = data['X_train']['TS'].unique()

    scores = []

    splits = KFold(n_splits=n_splits, random_state=0, shuffle=True).split(dates)
    for i, (local_train_date_ids, local_test_date_ids) in enumerate(splits):
        local_train_dates = dates[local_train_date_ids]
        local_test_dates = dates[local_test_date_ids]

        # Train 
        local_train_mask = (
            data['X_train']['ALLOCATION'].isin(allocations)
            & data['X_train']['TS'].isin(local_train_dates)
        )

        X_local_train = (
            data['X_train']
            .loc[local_train_mask, data['all_features']]
            .values
            .reshape(-1, len(allocations), len(data['all_features']))
        )

        y_local_train_bin = (
            (data['y_train'] > 0).astype(int)
            .loc[local_train_mask, 'target']
            .values
            .reshape(-1, len(allocations), 1)
        )

        # Test
        local_test_mask = (
            (data['X_train']['ALLOCATION'].isin(allocations)) &
            (data['X_train']['TS'].isin(local_test_dates))
        )

        X_local_test = (
            data['X_train']
            .loc[local_test_mask, data['all_features']]
            .values
            .reshape(-1, len(allocations), len(data['all_features']))
        )
        y_local_test_bin = (
            (data['y_train'] > 0).astype(int)
            .loc[local_test_mask, 'target']
            .values
            .reshape(-1, len(allocations), 1)
        )

        model = build_lstm_model(input_shape=(X_local_train.shape[1],X_local_train.shape[2]), lstm_units=param_grid['units'])
        
        model.fit(
                X_local_train, y_local_train_bin, 
                epochs=param_grid['epochs'], batch_size=param_grid['batch_size'], 
                validation_data=(X_local_test, y_local_test_bin), 
                callbacks=[tensorboard_callback],
                verbose=0
            )
        y_local_pred = model.predict(X_local_test)

        # print(y_local_pred)

        y_true = y_local_test_bin.reshape(-1)  # (n_samples * nb_alloc,)
        y_pred = (y_local_pred.reshape(-1) > 0.5).astype(int)
        score = accuracy_score(y_true, y_pred)
        scores.append(score)
    # mean_score = np.mean(scores)
   
    # # Entraîne le meilleur modèle sur l'ensemble des données d'entraînement pour cette allocation
    # X_local_best_train = data['X_train'].loc[train_idx,data['RET_features']]
    # y_local_best_train = data['y_train'].loc[train_idx,'target']

    # X_local_best_train = X_local_best_train.values.reshape((X_local_best_train.shape[0], X_local_best_train.shape[1], 1))

    # best_model = build_lstm_model(input_shape=(X_local_best_train.shape[1], 1), lstm_units=best_params['units'])
    # best_model.fit(
    #     X_local_best_train,
    #     y_local_best_train,
    #     epochs=best_params['epochs'],
    #     batch_size=best_params['batch_size'],
    #     callbacks=[tensorboard_callback],
    #     verbose=0
    # )

    # print(f"\nMeilleurs paramètres : {best_params} avec une accuracy de {best_score*100:.4f}%")
    # return best_score, best_model, best_params
    return np.mean(scores), model, param_grid


def save_model(model, path):
    # Sauvegarde du modèle au format natif Keras
    if not path.endswith('.keras'):
        path += '.keras'
    model.save(path)

def load_model(path):
    return load_model(path)

def main():
    data = import_and_prepoccess_data()
    
    # best_models = {}
    # best_scores = {}
    # best_params = {}
    
    nb_alloc = -1
    allocations = data['X_train']['ALLOCATION'].unique()[:nb_alloc]

    best_score, best_model, best_params = optimize(data=data, 
                                                allocations=allocations)
    # best_models[allocation] = best_model
    # best_scores[allocation] = best_score
    # best_params[allocation] = best_params
    # print(f"Allocation: {allocation}, Best Accuracy: {best_score*100:.4f}%, Best Params: {best_params}")
    # save_model(best_model, f'models/lstm_model_allocation_{allocation}.keras')
    print(f"For N°Allocations: {nb_alloc}, Best Accuracy: {best_score*100:.4f}%, Best Params: {best_params}")

if __name__ == "__main__":
    main()