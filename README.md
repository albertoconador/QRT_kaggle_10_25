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

