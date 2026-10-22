# Comparer des architectures, pas seulement des scores

## Question et hypothese

Sur les chiffres manuscrits de Kaggle Digit Recognizer, dans quelle mesure
la structure convolutionnelle ameliore-t-elle la classification par rapport
a un modele lineaire et a un MLP, pour un meme budget d'epoques ?

Hypothese : le CNN exploite mieux la structure spatiale que les images aplaties.
Cette comparaison ne mesure pas le meilleur score atteignable par chaque famille.

## Protocole fixe avant l'evaluation

- Source : `train.csv` de [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data).
- Separation stratifiee 60 % entrainement, 20 % validation, 20 % test interne.
- Graine du decoupage : 20261015. Memes indices pour toutes les architectures.
- Pixels divises par 255, sans augmentation ni normalisation apprise.
- Lineaire softmax : 784 -> 10 ; MLP : 784 -> 128 -> 10 avec ReLU ; CNN : architecture existante.
- Adam, taux 0.001, batch 128, 8 epoques maximum, CPU, 2 threads.
- Graines d'entrainement : 17, 42, 73. Le decoupage ne change pas entre les graines.
- Pour chaque execution, checkpoint de perte de validation minimale.
- Architecture retenue selon l'accuracy moyenne de validation ; perte moyenne pour departager.
- Pas de dropout ni recherche d'hyperparametres dans cette premiere experience.
- Evaluation finale des trois architectures seulement apres la selection, via une commande distincte.

L'accuracy et le F1 macro sont rapportes avec leur moyenne et leur ecart-type
entre graines. Cet ecart-type n'est pas un intervalle de confiance, ni une
mesure de variabilite entre populations. La perte et les courbes d'apprentissage
permettent d'observer le surapprentissage. Nombre de parametres et temps CPU
completent la comparaison ; un nombre egal d'epoques n'implique pas un cout egal.

## Execution

Depuis la racine du depot, apres installation des dependances :

```bash
python -m src.experiment train --train-csv data/train.csv --output artifacts/study
python -m src.experiment evaluate --train-csv data/train.csv --output artifacts/study
python -m unittest discover -s tests
```

La premiere commande ne calcule aucun score de test. La seconde exige les
memes donnees et les memes sources, verifiees par SHA-256. Un dossier existant
n'est pas ecrase. Les poids restent locaux dans `artifacts/`.

`protocol.json` conserve les reglages et versions ; `splits.npz` les indices ;
`history.csv` les epoques ; `validation.csv` la selection ; `test-metrics.csv`
et `test-summary.csv` l'evaluation finale. Les exemples d'erreurs sont les
douze premiers du modele retenu avec la premiere graine prevue, pas une
selection des exemples les plus flatteurs.

## Limites

Ce test interne provient d'un jeu deja utilise dans des travaux anterieurs :
il est reserve dans cette nouvelle procedure, mais n'est pas une validation
externe sur de nouvelles images. Le `test.csv` Kaggle est non etiquete et ne
sert pas a calculer nos metriques. Aucun score de classement Kaggle n'est revendique.

Trois graines ne suffisent pas a etablir une superiorite universelle. Le
modele lineaire est entraine avec Adam a budget fixe, pas optimise jusqu'a
convergence avec un solveur de regression logistique. Les performances ne
garantissent rien sur des photos, une autre ecriture ou un contexte de production.
Les resultats peuvent varier selon les versions et le materiel.

Une prochaine experience pourra isoler l'effet du dropout sur le MLP. Elle
devra etre declaree comme nouvelle analyse : le test deja consulte ne pourra
pas etre presente comme un test encore intact.

## References techniques

- [Reproductibilite dans PyTorch](https://docs.pytorch.org/docs/2.8/notes/randomness.html).
- [Separation des donnees et fuites d'information](https://scikit-learn.org/stable/common_pitfalls.html).
