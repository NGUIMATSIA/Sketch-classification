# Sketch Classification Project

## Overview
L'objectif de ce projet est de proposer un modèle de machine learning pour la classification de croquis à partir de la base de données [TU Berlin](https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/). Nous avons divisé cette base en ensembles d'entraînement (70%), de validation (15%) et de test (15%) en utilisant le notebook **Split_and_Observe**. Vous pouvez télécharger cette division via le lien suivant:
1. [Dataset](https://drive.google.com/drive/folders/1BTFb0hxsmjmgRxxdzVvDOOnGLxe3qSJa?usp=sharing)

Pour reproduire les résultats de ce travail, assurez-vous de modifier les chemins vers les dossiers dans les modules `validation.py` (ligne 66, 130, 99) et `main.py` (ligne 67). Nous avons ajouté des commentaires sur ces différentes lignes. Le notebook `experiment.ipynb` appelle tous les autres modules pour entraîner tous les modèles. Nous avons également sauvegardé les poids de tous les modèles entraînés, que vous pouvez télécharger en utilisant le lien suivant:
1. [Trained models](https://drive.google.com/drive/folders/1BTFb0hxsmjmgRxxdzVvDOOnGLxe3qSJa?usp=sharing)


