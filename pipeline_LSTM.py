import pandas as pd
import numpy as np
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Dropout
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from itertools import product

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
    }

def build_lstm_model(input_shape, lstm_units=50):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=lstm_units, return_sequences=False))
    model.add(Dropout(0.2))
    # model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# model = Sequential()
# model.add(Input(shape=(1, X_local_train.shape[2])))  # Ajout de la couche Input
# model.add(LSTM(units=n))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')

def optimize(data, allocation, n_splits=5):
    param_grid = {
        'units': [8, 16, 32],
        'epochs': [10, 20, 30],
        'batch_size': [16, 32, 64],
        # 'dense_units': [8, 16, 32]
    }

    n_iter = 10  # nombre de combinaisons à tester

    all_combinations = list(product(
        param_grid['units'], 
        param_grid['epochs'], 
        param_grid['batch_size']
    ))

    # tirage aléatoire
    sampled_combinations = random.sample(all_combinations, n_iter)

    best_score = 0
    best_model = None
    best_params = None
    train_idx = data['X_train'][data['X_train']['ALLOCATION'] == allocation].index


    # for units, epochs, batch_size, dense_units in product(param_grid['units'], param_grid['epochs'], param_grid['batch_size'], param_grid['dense_units']):
    # for units, epochs, batch_size in product(param_grid['units'], param_grid['epochs'], param_grid['batch_size']):
    for units, epochs, batch_size in sampled_combinations:
        scores = []
        splits = KFold(n_splits=n_splits, shuffle=True).split(train_idx)
        for i, (local_train_idx, local_test_idx) in enumerate(splits):
            dx,dy = data['X_train'].loc[local_train_idx,data['RET_features']].shape

            X_local_train = data['X_train'].loc[local_train_idx,data['RET_features']]
            y_local_train = data['y_train'].loc[local_train_idx,'target']
            X_local_test = data['X_train'].loc[local_test_idx,data['RET_features']]
            y_local_test = data['y_train'].loc[local_test_idx,'target']

            X_local_train_ret = data['X_train'].loc[local_train_idx,data['RET_features']].values.reshape((dx, dy, 1))
            X_local_train_vol = data['X_train'].loc[local_train_idx,data['SIGNED_VOLUME_features']].values.reshape((dx, dy, 1))
            X_local_train = np.concatenate((X_local_train_ret, X_local_train_vol), axis=2)

            X_local_test_ret = data['X_train'].loc[local_test_idx,data['RET_features']].values.reshape((X_local_test.shape[0], X_local_test.shape[1], 1))
            X_local_test_vol = data['X_train'].loc[local_test_idx,data['SIGNED_VOLUME_features']].values.reshape((X_local_test.shape[0], X_local_test.shape[1], 1))
            X_local_test = np.concatenate((X_local_test_ret, X_local_test_vol), axis=2)
            
            # model = build_lstm_model(input_shape=(X_local_train.shape[1],1), lstm_units=units, dense_units=dense_units)
            model = build_lstm_model(input_shape=(X_local_train.shape[1],2), lstm_units=units)
            
            model.fit(
                    X_local_train, y_local_train, 
                    epochs=epochs, batch_size=batch_size, 
                    validation_data=(X_local_test, y_local_test), 
                    callbacks=[tensorboard_callback],
                    verbose=0
                )
            y_local_pred = model.predict(X_local_test)

            
            score = accuracy_score((y_local_test>0).astype(int),
                        (y_local_pred>0).astype(int))
            scores.append(score)
        mean_score = np.mean(scores)
        # print(f"units={units}, epochs={epochs}, batch_size={batch_size}, dense_units= {dense_units}=> Accuracy: {mean_score*100:.4f}%")
        print(f"units={units}, epochs={epochs}, batch_size={batch_size} => Accuracy: {mean_score*100:.4f}%")
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'units': units, 'epochs': epochs, 'batch_size': batch_size} #, 'dense_units': dense_units}
    
    # Entraîne le meilleur modèle sur l'ensemble des données d'entraînement pour cette allocation
    X_local_best_train = data['X_train'].loc[train_idx,data['RET_features']]
    y_local_best_train = data['y_train'].loc[train_idx,'target']

    X_local_best_train = X_local_best_train.values.reshape((X_local_best_train.shape[0], X_local_best_train.shape[1], 1))

    best_model = build_lstm_model(input_shape=(X_local_best_train.shape[1], 1), lstm_units=best_params['units'])
    best_model.fit(
        X_local_best_train,
        y_local_best_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        callbacks=[tensorboard_callback],
        verbose=0
    )

    print(f"\nMeilleurs paramètres : {best_params} avec une accuracy de {best_score*100:.4f}%")
    return best_score, best_model, best_params


def save_model(model, path):
    # Sauvegarde du modèle au format natif Keras
    if not path.endswith('.keras'):
        path += '.keras'
    model.save(path)

def load_model(path):
    return load_model(path)

def main():
    data = import_and_prepoccess_data()
    
    best_models = {}
    best_scores = {}
    best_params = {}

    for allocation in data['X_train']['ALLOCATION'].unique()[:10]:
    
        best_score, best_model, best_params = optimize(data=data, 
                                                    allocation=allocation)
        best_models[allocation] = best_model
        best_scores[allocation] = best_score
        best_params[allocation] = best_params
        print(f"Allocation: {allocation}, Best Accuracy: {best_score*100:.4f}%, Best Params: {best_params}")
        save_model(best_model, f'models/lstm_model_allocation_{allocation}.keras')    

if __name__ == "__main__":
    main()