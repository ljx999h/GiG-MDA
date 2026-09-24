# Reproduction record: cold-drug XGBoost predictions and device sensitivity

This directory records a reproduction check of the four main XGBoost
configurations in the cold-drug evaluation, together with a comparison of the
GPU and CPU implementations of XGBoost's histogram method.

## What is here

```
predictions_gpu/   per-pair predictions produced with device=cuda (the device used for all reported values)
predictions_cpu/   per-pair predictions produced with device=cpu
```

Each prediction file contains one row per held-out candidate pair:

| column      | meaning                                              |
|-------------|------------------------------------------------------|
| `drugID`    | drug identifier as used in the split manifest        |
| `diseaseID` | disease identifier as used in the split manifest     |
| `label`     | true label (1 = verified association, 0 = unverified)|
| `score`     | predicted probability of the positive class          |

Four configurations are dumped per dataset/seed: `MiRAGE` (GBA features only),
`MiRAGE+MolEmb32`, `MiRAGE+embed` (GRMF embeddings), and
`MiRAGE+embed+MolEmb32` (three-channel). The accompanying `*_summary.csv`
files report AUROC, average precision (AP), and AP recomputed with an
independent implementation.

Datasets and seeds covered: C-Dataset, cold-drug splits, seeds 42 and 7.

## How to recompute

```bash
python code/dump_cold_predictions.py --dataset C --seed 42 --out-dir reproduction/predictions_gpu
```

Average precision can be recomputed from any prediction file without the
project code:

```python
import pandas as pd
from sklearn.metrics import average_precision_score
d = pd.read_csv('reproduction/predictions_gpu/C_s42_MiRAGE_embed_MolEmb32_pred.csv')
print(average_precision_score(d.label, d.score))
```

## Result of the check

| configuration           | device=cuda | device=cpu | manuscript value |
|-------------------------|-------------|------------|------------------|
| C seed 42, MiRAGE       | 0.3003      | 0.3116     | 0.3003           |
| C seed 42, +MolEmb32    | 0.3005      | 0.3175     | 0.3005           |
| C seed 42, +GRMF        | 0.3046      | 0.2710     | 0.3046           |
| C seed 42, three-channel| 0.3015      | 0.3047     | 0.3015           |
| C seed 7,  MiRAGE       | 0.1269      | 0.0992     | 0.1269           |
| C seed 7,  +MolEmb32    | 0.2287      | 0.2482     | 0.2287           |
| C seed 7,  +GRMF        | 0.2914      | 0.2703     | 0.2914           |
| C seed 7,  three-channel| 0.2264      | 0.2734     | 0.2264           |

(AUPR values.)

Under `device=cuda` the released code reproduces the reported values for all
four configurations in both splits. Under `device=cpu` the same code, splits,
features, and training negatives give different values, and in the seed-42
split the order of the baseline and the three-channel configuration is
reversed. XGBoost's histogram method builds its bins and accumulates gradients
in a different order on the two devices, so closely spaced configurations are
not expected to preserve their ordering across devices. All values in the
manuscript were produced with `device=cuda`.

## Inputs

The dumps were produced from the released split manifests together with:

- `code/build_mirage_features.py --neighbor-source r2train --manifest <cold manifest> --exclude MolFormer`
  (training-local GBA features for the cold split)
- `code/pretrain_gigs_split.py --manifest <cold manifest>` (training-local GRMF embeddings)
- `code/negative_mining_oof.py --manifest <cold manifest> --score <features>` (committee-filtered negatives)

XGBoost 3.0.2, scikit-learn 1.7.2, Python 3.12.7, CUDA device: NVIDIA GeForce
RTX 4060 Laptop GPU. The CPU dumps were produced on the same machine and the
same package versions.
