# fichier: app_gradio.py
import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

path = 'data/'

def FAR(row):
    ret = np.array(row['ret_ts'])
    vol = np.array(row['vol_ts'])
    return np.sum(ret * vol) / np.sum(np.abs(vol))


def load_data():


    X_train = pd.read_csv(path + 'X_train.csv',index_col='ROW_ID')
    X_test = pd.read_csv(path + 'X_test.csv',index_col='ROW_ID')

    y_train = pd.read_csv(path + 'y_train.csv',index_col='ROW_ID')
    sample_submission = pd.read_csv(path + 'sample_submission.csv',index_col='ROW_ID')

    RET_features = [f'RET_{i}' for i in range(1,21)]
    SIGNED_VOLUME_features = [f'SIGNED_VOLUME_{i}' for i in range(1,21)]
    TURNOVER_features = ['AVG_DAILY_TURNOVER']

    RET_features_subset = []
    for i in [3,5,10,15,20]:
        RET_features_subset.append(f'RET_{i}')

    data_explore = X_train.copy()
    data_explore['TARGET'] = y_train['target']
    data_explore['ret_ts'] = data_explore[RET_features].values.tolist()
    data_explore['vol_ts'] = data_explore[SIGNED_VOLUME_features].values.tolist()

    data_explore['category'] = data_explore['TARGET'].apply(lambda x: 1 if x > 0 else 0)

    data_explore['FAR'] = data_explore.apply(FAR, axis=1)
    return data_explore


def plot_far_distribution(nb_alloc, bins=30, show_kde=True, show_target=True):
    # Création du graphique
    data = load_data()
    fig, ax = plt.subplots(figsize=(10,6))
    if nb_alloc < 10:
        allocation_label = f'ALLOCATION_0{nb_alloc}'
    else:
        allocation_label = f'ALLOCATION_{nb_alloc}'
    
    if show_target:
        sns.histplot(
            data=data[data['ALLOCATION'] == allocation_label],
            x='FAR',
            hue=data['TARGET'] > 0,
            kde=show_kde,
            bins=bins,
            legend=True,
            palette=["red", "green"]
        )
    else:
        sns.histplot(
            data=data[data['ALLOCATION'] == allocation_label],
            x='FAR',
            kde=show_kde,
            bins=bins,
            legend=True,
            color="blue"
            )
    ax.set_title(f"Distribution allocation n°{nb_alloc} de FAR selon TARGET > 0")
    ax.set_xlabel("FAR")
    ax.set_ylabel("Count")
    
    return fig

# --- Interface Gradio ---
interface = gr.Interface(
    fn=plot_far_distribution,
    inputs=[
        gr.Slider(minimum=1, maximum=46, value=1, step=1, label="allocation"),
        gr.Slider(minimum=10, maximum=100, value=30, label="Nombre de bins"),
        gr.Checkbox(label="Afficher KDE", value=True),
        gr.Checkbox(label="Afficher avec target", value=True)
    ],
    outputs=gr.Plot(),
    title="Distribution de FAR selon TARGET",
    description="Affiche l'histogramme de FAR en séparant selon TARGET > 0"
)

interface.launch()
