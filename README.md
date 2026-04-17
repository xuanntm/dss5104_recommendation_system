# DSS5104 Recommendation System

A comparative study of recommendation system algorithms for NUS DSS5104, evaluating classical and deep learning approaches on explicit and implicit feedback datasets.

---

## Project Overview

This repository implements and evaluates the following recommendation models:

| Model | Type | Dataset |
|---|---|---|
| Popularity Baseline | Heuristic | MovieLens 100K, LastFM-2K |
| SVD (Matrix Factorization) | Classical | MovieLens 100K (explicit) |
| ALS (Alternating Least Squares) | Classical | LastFM-2K (implicit) |
| NCF (Neural Collaborative Filtering) | Deep Learning | MovieLens 100K |
| Two-Tower | Deep Learning | LastFM-2K and MovieLens 100K |

**Evaluation metrics:** `hr@10`, `ndcg@10`, `map@10`.

---

## Repository Structure

```
dss5104_recommendation_system/
├── data/
│   ├── raw/
│   │   ├── ml-100k/                # MovieLens 100K raw data
│   │   │   ├── README
│   │   │   └── u.data
│   │   └── lastfm-2K/              # LastFM-2K raw data
│   │       ├── readme.txt
│   │       └── user_artists.dat
│   └── processed/
│       ├── ml-100k/                # Processed MovieLens splits
│       │   ├── train.parquet
│       │   ├── val.parquet
│       │   ├── test.parquet
│       │   └── meta.pkl
│       └── lastfm-2k/              # Processed LastFM splits
│           ├── train.parquet
│           ├── val.parquet
│           ├── test.parquet
│           └── meta.pkl
├── src/
│   ├── 01_data_preprocessing.ipynb  # Data loading, binarization, filtering, splitting
│   ├── 02_baseline.ipynb            # Popularity, SVD, ALS baselines
│   ├── 03_ncf.ipynb                 # Neural Collaborative Filtering
│   └── 04_two_tower.ipynb           # Two-Tower retrieval model
├── requirements.txt
└── README.md
```

---

## Datasets

### MovieLens 100K — Explicit Feedback
- 100,000 ratings from 943 users on 1,682 movies
- Ratings ≥ 4 are binarized as positive
- Filtered with a 5-core threshold
- Processed splits saved under `data/processed/ml-100k/`

### LastFM-2K — Implicit Feedback
- Listening records for 1,892 users and 2,823 artists
- Implicit interactions derived from play counts
- Filtered with a 5-core threshold
- Processed splits saved under `data/processed/lastfm-2k/`

Both datasets use per-user chronological splits: 80% train, 10% validation, 10% test.

---

## Setup

### Prerequisites
- Python 3.12
- `venv` or `conda`

### Installation

```bash
git clone <repo-url>
cd dss5104_recommendation_system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Main Dependencies

- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `pyarrow`
- `torch`
- `scikit-surprise`
- `implicit`
- `matplotlib`
- `seaborn`
- `tqdm`

---

## Running the Notebooks

Open and execute the notebooks in order from the `src/` directory:

```bash
cd src
jupyter notebook
```

| Notebook | Purpose |
|---|---|
| `01_data_preprocessing.ipynb` | Preprocess raw data, binarize signals, apply k-core filtering, remap IDs, and save train/val/test splits |
| `02_baseline.ipynb` | Evaluate popularity baseline; train and tune SVD on MovieLens; train and tune ALS on LastFM-2K |
| `03_ncf.ipynb` | Train and evaluate Neural Collaborative Filtering on MovieLens 100K |
| `04_two_tower.ipynb` | Train and evaluate Two-Tower models on LastFM-2K and compare on MovieLens |

> Each notebook depends on the previous notebook's outputs.

---

## Model Details

### NCF — Neural Collaborative Filtering
Implements the GMF + MLP fusion architecture from He et al. (2017):
- GMF branch: element-wise product of user/item embeddings
- MLP branch: stacked dense layers on concatenated embeddings
- Final score is produced by combining both branches

### Two-Tower
A retrieval-focused model with separate user and item encoders:
- User and item embeddings are learned independently
- Similarity is computed with dot product after embedding normalisation
- Suitable for scalable retrieval and embedding-based ranking

### Preprocessing Pipeline
- Binarize explicit ratings and convert implicit play counts to positive events
- Apply 5-core filtering to ensure at least 5 interactions per user/item
- Split data chronologically at the user level to avoid future leakage
- Evaluate cold-start and regular users separately

---

## Artifacts

Saved outputs include:
- `data/processed/ml-100k/`
- `data/processed/lastfm-2k/`
- `data/processed/baseline_results.csv`
- `data/processed/final_results.csv`
- `data/processed/ncf_movielens.pt`
- `data/processed/two_tower_lastfm.pt`
- `data/processed/two_tower_movielens.pt`

---

## References

- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. WWW 2017.
- MovieLens dataset: https://grouplens.org/datasets/movielens/
- LastFM-2K dataset: Cantador et al., HetRec 2011 / RecSys 2011.
