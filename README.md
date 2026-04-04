# DSS5104 Recommendation System

A comparative study of recommendation system algorithms for NUS DSS5104, evaluating classical and deep learning approaches on two real-world datasets across explicit and implicit feedback settings.

---

## Project Overview

This project implements and compares four recommendation models:

| Model | Type | Dataset |
|---|---|---|
| Popularity Baseline | Heuristic | Both |
| SVD (Matrix Factorization) | Classical | MovieLens (explicit) |
| ALS (Alternating Least Squares) | Classical | LastFM-2K (implicit) |
| NCF (Neural Collaborative Filtering) | Deep Learning | MovieLens |
| Two-Tower | Deep Learning | Both |

**Evaluation metrics:** HR@10, NDCG@10, MAP@10 — with cold-start vs. regular user breakdown.

---

## Repository Structure

```
dss5104_recommendation_system/
├── data/
│   ├── raw/
│   │   ├── ml-100k/               # MovieLens 100K (explicit ratings)
│   │   │   └── u.data
│   │   └── lastfm-2K/             # LastFM-2K (implicit listening counts)
│   │       └── user_artists.dat
│   └── processed/
│       ├── movielens_1m/          # Processed MovieLens splits
│       │   ├── train.parquet
│       │   ├── val.parquet
│       │   ├── test.parquet
│       │   └── meta.pkl
│       └── lastfm_2k/             # Processed LastFM splits
│           ├── train.parquet
│           ├── val.parquet
│           ├── test.parquet
│           └── meta.pkl
├── src/
│   ├── 01_data_preprocessing.ipynb   # Binarization, k-core filtering, temporal split
│   ├── 02_baseline.ipynb             # Popularity, SVD, ALS baselines
│   ├── 03_ncf.ipynb                  # Neural Collaborative Filtering
│   └── 04_two_tower.ipynb            # Two-Tower retrieval model
├── requirements.txt
└── README.md
```

---

## Datasets

### MovieLens 100K — Explicit Feedback
- 100,000 ratings from 943 users on 1,682 movies
- Binarized: ratings ≥ 4 treated as positive (label=1)
- After 5-core filtering: ~54K interactions, 938 users, 1,008 items

### LastFM-2K — Implicit Feedback
- 92,834 user-artist listening records from 1,892 users
- Play counts used as proxy for temporal ordering
- After 5-core filtering: ~71K interactions, 1,859 users, 2,823 items

Both datasets are split **per-user chronologically** (80% train / 10% val / 10% test) to avoid data leakage.

---

## Setup

### Prerequisites
- Python 3.12
- `venv` or `conda`

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd dss5104_recommendation_system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas, numpy, scipy, scikit-learn
pyarrow          # Parquet I/O
torch            # NCF and Two-Tower models
scikit-surprise  # SVD
implicit         # ALS
matplotlib
tqdm
```

---

## Running the Notebooks

Run the notebooks in order from the `src/` directory:

```bash
cd src
jupyter notebook
```

| Notebook | Description |
|---|---|
| `01_data_preprocessing.ipynb` | Load raw data, binarize, k-core filter, remap IDs, temporal split, save to `data/processed/` |
| `02_baseline.ipynb` | Popularity baseline; SVD with hyperparameter tuning (MovieLens); ALS with hyperparameter tuning (LastFM-2K) |
| `03_ncf.ipynb` | Neural Collaborative Filtering (GMF + MLP branches); negative sampling sensitivity analysis |
| `04_two_tower.ipynb` | Two-Tower model with separate user/item encoders; embedding PCA visualisation |

> Each notebook requires the previous one to have been run first.

---

## Model Details

### NCF — Neural Collaborative Filtering
Implements [He et al., 2017](https://arxiv.org/abs/1708.05031). Two parallel branches:
- **GMF branch**: element-wise product of user and item embeddings
- **MLP branch**: concatenated embeddings through a stack of linear layers
- Outputs fused and projected to a scalar score via binary cross-entropy loss

### Two-Tower
Separate user and item encoder networks (embedding + MLP tower). Scored via cosine similarity after L2 normalisation. Key advantage: item embeddings can be pre-computed offline for fast ANN retrieval at serving time.

### Preprocessing Pipeline
- **Binarization**: explicit ratings ≥ 4 → positive; play counts → presence
- **k-core filtering** (k=5): iteratively removes users/items with fewer than 5 interactions
- **Temporal split**: per-user chronological 80/10/10 split to prevent leakage
- **Cold-start flagging**: test users with < 5 training interactions are flagged

---

## References

- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). *Neural collaborative filtering.* WWW 2017.
- MovieLens dataset: [grouplens.org](https://grouplens.org/datasets/movielens/)
- LastFM-2K dataset: Cantador et al., HetRec 2011 @ RecSys 2011.
