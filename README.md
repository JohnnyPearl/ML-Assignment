# Comparative Study of Machine Learning and Deep Learning for Hate Speech Classification

**Course:** Machine Learning (CO3117)
**Semester:** II, Academic Year 2025–2026
**Department:** Computer Science, Ho Chi Minh City University of Technology, VNU-HCM

## Instructor

- **Trương Vĩnh Lân**

## Team Members (Group 2)

| Name | Student ID | Email |
|------|-----------|-------|
| Trần Quốc Bảo Long      | 2252453 | <long.tran041102@hcmut.edu.vn>     |
| Nguyễn Bình Nguyên      | 2252545 | <nguyen.nguyenbinh1509@hcmut.edu.vn>   |
| Đặng Ngọc Phú           | 2252617 | <phu.dangngoc@hcmut.edu.vn>      |
| Đặng Duy Nguyên         | 2352821 | <nguyen.dangcolece@hcmut.edu.vn>|
| Nguyễn Văn Hoàng Phát   | 2352896 | <phat.nguyennvhp010405@hcmut.edu.vn>     |


## Project Objectives

This project develops and evaluates models for detecting hate speech in short
social-media texts. We conduct a **comparative study** between traditional
machine learning approaches (Multinomial NB, Gaussian NB, Logistic Regression,
LinearSVC, k-NN, Random Forest, Gradient Boosting) and modern deep learning
methods (fine-tuned DistilBERT, BiLSTM with GloVe), assessing their
effectiveness and trade-offs on a binary toxicity-classification task using
the [Twitter Toxic Tweets](https://www.kaggle.com/datasets/umitka/twitter-toxic-tweets)
dataset (31,962 tweets, 93%/7% imbalance).

The full experimental matrix produces:

- **27 baseline experiments** (7 classifiers × 5 feature representations, with
  compatibility filtering)
- **3 hyperparameter-tuned variants** (top performers, GridSearchCV)
- **2 deep-learning models** (fine-tuned DistilBERT and BiLSTM with GloVe)
- **9 SMOTE-vs-class-weight comparisons** (3 models × 3 sparse features)

**Best result:** Fine-tuned DistilBERT achieves F1-macro = 0.871, 3.5
percentage points above the best traditional ML model (LinearSVC + TF-IDF at
F1-macro = 0.836).

## Folder Structure

```
ML-Assignment/
├── notebooks/
│   └── hate_speech_classification.ipynb   # Main pipeline (102 cells, sections 0-13)
├── modules/
│   ├── preprocessing.py                   # Text cleaning (dual variant)
│   ├── feature_extraction.py              # 5 feature methods
│   ├── model_training.py                  # 7 classifiers, grid runner, tuning, viz
│   └── utils.py                           # EDA plotting helpers
├── reports/
│   ├── ML_Final_Report.pdf                # Compiled final report (PDF)
│   ├── ML_Report_1.pdf                    # Progress Report 1 (topic + data)
│   ├── ML_Report_2.pdf                    # Progress Report 2 (EDA + features)
│   ├── ML_Report_3.pdf                    # Progress Report 3 (model training + DL)
│   └── ML_Report_4.pdf                    # Progress Report 4 (extended analysis)
├── features/                              # 12 saved feature files (.npz / .npy)
│   ├── bow_train.npz                      # Bag-of-Words (sparse, 10K dims)
│   ├── bow_test.npz                       # Bag-of-Words test subset
│   ├── tfidf_train.npz                    # TF-IDF features (sparse, 15K dims)
│   ├── tfidf_test.npz                     # TF-IDF test subset
│   ├── char_tfidf_train.npz               # Character-level TF-IDF (sparse, 10K dims)
│   ├── char_tfidf_test.npz                # Character-level TF-IDF test subset
│   ├── glove_train.npy                    # GloVe embeddings (dense, 200 dims)
│   ├── glove_test.npy                     # GloVe test subset
│   ├── distilbert_train.npy               # DistilBERT embeddings (dense, 768 dims)
│   ├── distilbert_test.npy                # DistilBERT test subset
│   ├── y_train.npy                        # Train labels (0=non-toxic, 1=toxic)
│   └── y_test.npy                         # Test labels (0=non-toxic, 1=toxic)               
├── datas/
│   └── twitter_toxic_tweets.csv           # Raw dataset (31,962 tweets)
├── glove_twitter/                         # (auto-downloaded) GloVe vectors
├── README.md                              # This file
└── requirements.txt                       # Python dependencies
```

## Feature Files

The `features/` directory contains pre-extracted feature representations for both training and test sets:

- **Sparse Features** (stored as `.npz` — compressed NumPy sparse matrices):
  - **BoW** (Bag-of-Words): 10K vocabulary size; count-based representation
  - **TF-IDF**: 15K vocabulary size; weighted by term frequency and inverse document frequency
  - **Char TF-IDF**: 10K character n-gram vocabulary; captures character-level patterns
  
- **Dense Features** (stored as `.npy` — NumPy arrays):
  - **GloVe**: 200-dimensional pre-trained word embeddings from Twitter corpus
  - **DistilBERT**: 768-dimensional contextual embeddings from fine-tuned DistilBERT `[CLS]` tokens

- **Labels** (`y_train.npy`, `y_test.npy`):
  - Binary labels: 0 = non-toxic, 1 = toxic
  - Class distribution: ~93% non-toxic, ~7% toxic

All features are split 80/20 (train/test) with stratification to preserve class distribution. Use these files directly in `modules/model_training.py` with the `load_features()` function.

## Setup Instructions

### Local environment (Windows / PowerShell)

```powershell
# 1. Create and activate virtual environment with uv
uv venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Install PyTorch (GPU - CUDA 12.4)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CPU fallback: uv pip install torch torchvision

# 4. Register Jupyter kernel
python -m ipykernel install --user --name ml-hate-speech --display-name "ML Hate Speech (uv)"

# 5. Launch the notebook
jupyter notebook notebooks/hate_speech_classification.ipynb
```

### Google Colab

The notebook auto-detects Colab. Open
`notebooks/hate_speech_classification.ipynb` in Colab and select
**Runtime → Change runtime type → GPU** before running. The dataset is
downloaded automatically from this repository.

## Data

- **Source:** [Twitter Toxic Tweets on Kaggle](https://www.kaggle.com/datasets/umitka/twitter-toxic-tweets)
- **Format:** CSV with three columns (`id`, `tweet`, `label`)
- **Size:** 31,962 tweets, 93% non-toxic / 7% toxic
- **Auto-download (Colab):** the notebook fetches the CSV from this repo's
  `datas/` directory via the GitHub raw URL
- **Auto-download (GloVe):** the notebook downloads
  `glove.twitter.27B.200d.txt` (~2 GB) from Stanford NLP on first run if it
  is not already present in `glove_twitter/`

## Run

The full pipeline lives in a single notebook. Execute cells top-to-bottom
(or `Kernel → Restart & Run All`):

1. **Sections 0–7** (CPU, ~5 min) — EDA, preprocessing, feature extraction
2. **Sections 8–9** (CPU, ~10 min) — 27 baseline experiments + GridSearchCV tuning
3. **Section 10** (GPU, ~20 min) — DistilBERT fine-tuning + BiLSTM training
4. **Sections 11–13** (CPU, ~2 min) — final comparison, error analysis, interpretability

If a GPU is unavailable, sections 0–9 and 11–13 still run on CPU; only
section 10 (deep-learning training) requires CUDA.

## Key Findings

| Approach                    | F1-macro | Train time          | Notes                       |
|-----------------------------|---------:|---------------------|-----------------------------|
| LinearSVC + TF-IDF          |   0.8358 | < 1 s               | Fastest; no GPU needed      |
| Random Forest + Char TF-IDF |   0.8335 | ~30 s               | Robust; interpretable       |
| BiLSTM + GloVe              |   0.8198 | ~5 min (GPU)        | Needs GloVe (2 GB)          |
| DistilBERT (fine-tuned)     | **0.8710** | ~15 min (GPU)     | Best quality                |

Detailed results, confusion matrices, ROC curves, error analysis, model
interpretability (top discriminative features), and SMOTE-vs-class-weight
comparison are documented in
[`reports/ML_Final_Report.pdf`](reports/ML_Final_Report.pdf).

## Links

- **GitHub repository:** <https://github.com/JohnnyPearl/ML-Assignment>
- **Final report (PDF):** [reports/ML_Final_Report.pdf](reports/ML_Final_Report.pdf)
- **Notebook:** [notebooks/hate_speech_classification.ipynb](notebooks/hate_speech_classification.ipynb)
- **Dataset:** <https://www.kaggle.com/datasets/umitka/twitter-toxic-tweets>

## License

For educational use only. Course project for Machine Learning (CO3117) at
Ho Chi Minh City University of Technology.
