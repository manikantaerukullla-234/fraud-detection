# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using Random Forest on the Kaggle Credit Card Fraud dataset.

## Results

| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.98 |
| Precision (Fraud) | ~0.93 |
| Recall (Fraud) | ~0.87 |
| F1 (Fraud) | ~0.90 |

## Project Structure

```
fraud-detection/
├── data/               # Place creditcard.csv here (not tracked by git)
├── src/
│   ├── preprocess.py   # Data loading, scaling, SMOTE
│   └── train.py        # Model training and evaluation
├── app.py              # Streamlit demo app
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Download creditcard.csv and place it in the data/ folder
```

## Train the Model

```bash
cd src
python train.py
```

This will print the classification report, save `roc_curve.png`, and save `model.pkl`.

## Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Dataset

- Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions · 492 fraud cases (0.17%)
- Features V1–V28 are PCA-transformed for privacy

## Tech Stack

- Python, Scikit-learn, imbalanced-learn
- Streamlit (demo UI)
- SMOTE for class imbalance handling

## How to Push to GitHub

```bash
git init
git add .
git commit -m "initial commit: fraud detection model"
git remote add origin https://github.com/yourusername/fraud-detection.git
git push -u origin main
```
