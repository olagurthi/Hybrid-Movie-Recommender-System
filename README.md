# 🎬 Hybrid Movie Recommender System

A complete implementation of a **Hybrid Recommender System** combining Content-Based Filtering and Collaborative Filtering, built on the MovieLens dataset.

-----

## 📐 Architecture (from lecture slides)

```
┌─────────────────────────────────────────────────────────┐
│                  HYBRID RECOMMENDER                     │
│                                                         │
│  ┌──────────────────┐      ┌────────────────────────┐  │
│  │  Content-Based   │      │  Collaborative         │  │
│  │  Filtering (CBF) │      │  Filtering (CF)        │  │
│  │                  │      │                        │  │
│  │ TF-IDF genres    │      │ SVD Matrix Factorize.  │  │
│  │ + year feature   │      │ + User-Based CF        │  │
│  │                  │      │                        │  │
│  │ i_j = TF-IDF(g)  │      │  R ≈ U Σ Vᵀ           │  │
│  │ u_i = Σ r·i_j    │      │                        │  │
│  └────────┬─────────┘      └───────────┬────────────┘  │
│           │                            │               │
│           └────────────┬───────────────┘               │
│                        ▼                               │
│          score = α·CBF + (1-α)·CF                      │
└─────────────────────────────────────────────────────────┘
```

|Component              |Method                                                                  |Lecture Slides|
|-----------------------|------------------------------------------------------------------------|--------------|
|**Content-Based (CBF)**|TF-IDF item profiles → rating-weighted user profiles → cosine similarity|25–34         |
|**Collaborative (CF)** |Truncated SVD (model-based) + User-Based CF (memory-based)              |37–39         |
|**Hybrid Fusion**      |`score = α · CBF + (1-α) · CF`                                          |24, 42        |
|**Evaluation**         |RMSE, Precision@K, Recall@K                                             |23            |

-----

## 📁 Files

```
├── hybrid_recommender.ipynb   # Main notebook (all code)
├── movies.csv                 # MovieLens dataset (movieId, title, genres)
├── README.md
└── outputs/
    ├── eda.png                # Exploratory analysis plots
    ├── user_0_profile.png     # Example user genre profile
    └── evaluation.png         # Precision/Recall vs alpha
```

-----

## ⚙️ Requirements

```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

Or all at once:

```bash
pip install pandas numpy scikit-learn scipy matplotlib jupyter
```

-----

## 🚀 How to Run

1. **Clone the repo**

```bash
git clone https://github.com/<your-username>/hybrid-movie-recommender.git
cd hybrid-movie-recommender
```

1. **Place `movies.csv` in the same folder as the notebook**
1. **Launch Jupyter**

```bash
jupyter notebook hybrid_recommender.ipynb
```

1. **Run all cells** — the notebook is self-contained and installs missing packages automatically.

-----

## 🧠 Key Concepts Implemented

### Utility Function (Lecture slide 18)

The user–item matrix **r : U × I → R** maps users and items to ratings (1–5 stars).  
Since `movies.csv` has no ratings, a realistic sparse matrix is simulated (500 users, ~2% density) with genre-biased preferences.

### Item Profiles (Slides 26–27)

Each movie is represented as a TF-IDF vector over its genres — words that are frequent in that movie but distinctive across the catalog score higher.

### User Profiles (Slides 28–32)

User preference vector built as a **rating-weighted average** of item profiles:

$$\mathbf{u}*i = \frac{\sum*{j \in \mathcal{I}_u} r(u,j) \cdot \mathbf{i}*j}{\sum*{j \in \mathcal{I}_u} r(u,j)}$$

### Content-Based Prediction (Slide 34)

Score for unseen item `i` for user `u`:

$$\text{score_CBF}(u, i) = \cos(\mathbf{u}_u,, \mathbf{i}_i)$$

### Collaborative Filtering — SVD (Slide 37)

Truncated SVD decomposes the mean-centered rating matrix:

$$R \approx U \Sigma V^T, \quad k=50 \text{ latent factors}$$

### Hybrid Fusion (Slides 24, 42)

```python
hybrid_score = alpha * cbf_score + (1 - alpha) * cf_score
```

- `alpha = 1.0` → Pure content-based (good for cold-start users)
- `alpha = 0.0` → Pure collaborative (good for users with rich history)
- `alpha = 0.5` → Balanced (default)

### Cold-Start Handling (Slide 42)

New users with no history get recommendations via `alpha=1.0` (pure CBF) using only their stated genre preferences.

-----

## 📊 Evaluation Metrics (Slide 23)

|Metric         |Description                                        |
|---------------|---------------------------------------------------|
|**RMSE**       |Root Mean Square Error on held-out ratings         |
|**MAE**        |Mean Absolute Error                                |
|**Precision@K**|Fraction of top-K recommendations that are relevant|
|**Recall@K**   |Fraction of relevant items that appear in top-K    |

-----

## 🔧 Tuning Alpha

```python
recommend_hybrid(user_id=0, n=10, alpha=0.5)
```

|Alpha|Label    |Best For                      |
|-----|---------|------------------------------|
|0.0  |Pure CF  |Active users with rich history|
|0.25 |CF-heavy |Users with moderate history   |
|0.5  |Balanced |Default / general use         |
|0.75 |CBF-heavy|Users with limited history    |
|1.0  |Pure CBF |Cold-start / new users        |

-----

## 📝 Notes

> This project was developed as part of **L18: Hands-On** for the Machine Learning course at UNYT.  
> Group implementing: **Hybrid approach** (Content-Based + Collaborative Filtering).
