# Données MNIST

Ce dossier doit contenir les fichiers CSV du challenge Kaggle Digit Recognizer :

- `train.csv`
- `test.csv`

Ces fichiers ne sont pas versionnés dans Git parce qu'ils sont volumineux et reproductibles depuis la source du challenge.

Structure attendue :

```text
data/train.csv
data/test.csv
```

`train.csv` contient la colonne `label` suivie de `pixel0` à `pixel783`. `test.csv` contient uniquement les pixels.
