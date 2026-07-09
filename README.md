# MNIST Digit Recognizer avec PyTorch

Projet de classification de chiffres manuscrits sur le dataset MNIST au format CSV Kaggle. Le dépôt contient une implémentation PyTorch propre d'un petit CNN, un split train/validation reproductible et une génération optionnelle de fichier de soumission.

## Objectif

Construire un pipeline simple mais crédible pour :

- charger les fichiers `train.csv` et `test.csv` du challenge Kaggle Digit Recognizer ;
- normaliser les pixels dans `[0, 1]` ;
- entraîner un réseau convolutionnel sur les images 28x28 ;
- mesurer l'accuracy sur un split de validation ;
- générer un fichier `submission.csv` pour les images non labellisées.

## Structure

```text
.
├── src/
│   └── mnist_cnn.py       # Pipeline PyTorch complet
├── tests/
│   └── test_mnist_cnn.py  # Tests légers
├── data/
│   └── README.md          # Instructions pour les données
├── artifacts/             # Modèles et soumissions générés localement
├── cnn.py                 # Wrapper de compatibilité
├── mnist.py               # Wrapper de compatibilité
└── requirements.txt
```

Les fichiers CSV Kaggle ne sont pas versionnés. Place-les localement dans `data/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Données

Télécharger les données du challenge Kaggle Digit Recognizer, puis placer les fichiers ainsi :

```text
data/train.csv
data/test.csv
```

`train.csv` doit contenir une colonne `label` suivie de `pixel0` à `pixel783`. `test.csv` contient uniquement les pixels.

## Entraînement

Lancer un entraînement court :

```bash
python mnist.py --epochs 3
```

Lancer un smoke test rapide sur un sous-ensemble :

```bash
python mnist.py --epochs 1 --max-rows 2000 --batch-size 64
```

Forcer le CPU :

```bash
python mnist.py --epochs 1 --cpu
```

## Soumission Kaggle

Si `data/test.csv` est présent, le script peut générer une soumission :

```bash
python mnist.py --epochs 5 --submission-out artifacts/submission.csv
```

Le modèle entraîné est sauvegardé par défaut dans :

```text
artifacts/mnist_cnn.pt
```

## Tests

```bash
python -m unittest discover -s tests
```

Les tests ne nécessitent pas les CSV complets. Les tests PyTorch sont ignorés automatiquement si PyTorch n'est pas installé.

## Notes de crédibilité

Ce dépôt ne prétend pas à un score Kaggle sans journal d'entraînement vérifiable. Les résultats doivent être reportés dans le README uniquement après exécution complète avec la configuration utilisée : seed, nombre d'epochs, taille du split validation, accuracy et éventuel score Kaggle.
