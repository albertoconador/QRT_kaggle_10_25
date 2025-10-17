# QRT_kaggle_10_25

Ce dépôt contient un petit projet d'exploration pour une compétition Kaggle (données dans le dossier `data/`).

## Objectif
Fournir un pipeline d'entraînement léger et des outils pour reproduire l'environnement Conda localement.

## Prérequis
- Windows (les commandes ci-dessous sont pour PowerShell)
- Conda (Anaconda ou Miniconda) installé et disponible dans le PATH

## Installation de l'environnement Conda
1. Ouvrez PowerShell.
2. Placez-vous dans le répertoire racine du projet.
3. Créez l'environnement Conda à partir du fichier `conda_env.yml`:

```powershell
conda env create -f .\QRT_kaggle_10_25\conda_env.yml
```

4. Activez l'environnement:

```powershell
conda activate qrt_env
```

## Structure du dépôt
- `data/` : jeux de données d'entrée (X_train, y_train, X_test, sample_submission, notebook de benchmark)

## Structure du code 
- Features :
    - Returns : Moyenne, Ecart-type, EVM, Streak, Entropie, Sharpe10, Sortino10, t-stat, Max drawdown et recovery
    - Volumes : Moyenne, Ecart-type, Autocorrelation, Correlation Returns-Liquidité
- Preprocessing :
    Utilisation de la structure de scikit-learn avec les méthodes fit et transform. Cela permet d'avoir une meilleur lisibilité et l'utilisation de Pipeline.

    - Winsorizer : Clip les valeurs extrêmes qui sont dans les quantiles extérieures.
    - GroupSVDEncoder : Embedding des colonnes features par la colonne group_col dans de nouvelles colonnes. 
    - DAE : Implémentation du DAE sont le format scikit-learn.

- Feature Selection:
    RobustFeatureSelector : Détection des features qui ont le plus d'importance et qui sont décorréllé des autres features.

- Model : 
    Pipeline d'entrainement :
    1. Ajout des features
    2. Filtrage des outliers et Standardization
    3. Pour n_splits, on embedde les datas avec SVD, puis on débruite avec le DAE, on sélectionne nos features les plus robustes puis on prédit les valeurs de validation et de test.

    Prédiction :
    La prédiction est avec l'aggrégation des prédictions fait par nos n_splits modèles.