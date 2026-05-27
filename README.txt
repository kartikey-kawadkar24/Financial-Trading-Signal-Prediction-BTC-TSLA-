================================================================================
README.txt
AIT 582 - Project 3: News-Driven Trading Signal Classification
================================================================================

--------------------------------------------------------------------------------
1. PROBLEM INTRODUCTION
--------------------------------------------------------------------------------

The goal of this project is to predict trading actions (BUY, HOLD, or SELL)
for two financial assets — Bitcoin (BTC) and Tesla (TSLA) — by combining
financial news sentiment with historical price data.

Traditional trading models rely only on numerical price indicators, ignoring
the impact that financial news and public sentiment can have on asset prices.
This project addresses that gap by building a multi-modal classification system
that fuses:
  - Structured price/market features (moving averages, volatility, returns)
  - VADER sentiment scores computed from news headlines
  - FinBERT deep-learning sentiment embeddings and probabilities

The core problem:
  Given a day's news articles and market data for BTC or TSLA, predict
  whether the correct trading action for the next period is BUY, HOLD, or SELL.

This is a supervised multi-class classification problem with three output labels:
  - BUY   → model predicts the asset price will rise
  - HOLD  → model predicts no significant movement
  - SELL  → model predicts the asset price will fall

--------------------------------------------------------------------------------
2. SOLUTION OUTLINE (Point-by-Point)
--------------------------------------------------------------------------------

Step 1: Load Dataset
  - Load the TheFinAI/CLEF_Task3_Trading dataset from HuggingFace.
  - The dataset contains two splits: BTC and TSLA.
  - Merge both into a single unified DataFrame.

Step 2: Pre-processing
  - Convert the list of daily news articles into a single concatenated
    string column called 'news_text'.
  - Convert 'future_price_diff' to numeric and handle type errors.
  - Drop columns with excessive missing values ('10k', '10q').
  - Fill remaining missing values (median imputation for numeric columns).
  - Reset and sort the index by asset and date.

Step 3: Feature Engineering
  - Compute momentum features:
      * ret_1, ret_3, ret_7  → 1-day, 3-day, 7-day percentage returns
  - Compute volatility features:
      * vol_5, vol_10        → 5-day and 10-day rolling standard deviation
  - Compute trend indicators:
      * ma_5, ma_10          → 5-day and 10-day moving averages
      * trend                → difference between ma_5 and ma_10
  - Compute derived interaction features:
      * momentum_strength    → return × volatility
      * return_to_vol        → return divided by volatility (risk-adjusted)
      * text_len             → character length of the news text
      * sentiment_change     → change in VADER sentiment from prior period

Step 4: VADER Sentiment Analysis
  - Use NLTK's SentimentIntensityAnalyzer to compute a compound sentiment
    score for each day's news text.
  - Scores range from -1 (very negative) to +1 (very positive).
  - Store result in 'sentiment' column.

Step 5: FinBERT Sentiment Analysis
  - Use the HuggingFace pipeline with model "yiyanghkust/finbert-tone".
  - Compute a signed confidence score:
      * positive label → +score
      * negative label → -score
      * neutral label  → 0
  - Store result in 'finbert_sentiment' column.

Step 6: FinBERT Embeddings
  - Load the ProsusAI/finbert model to extract 768-dimensional embeddings
    from news text using mean pooling over the last hidden state.
  - Apply PCA to reduce embeddings to 20 dimensions for efficiency.
  - Concatenate PCA embeddings with FinBERT class probabilities
    (positive, neutral, negative) for the final feature matrix.

Step 7: Final Feature Matrix
  - Combine three feature groups using np.hstack:
      (a) Scaled numeric features (market/price indicators + VADER sentiment)
      (b) PCA-reduced FinBERT embeddings (20 dimensions)
      (c) FinBERT class probabilities (3 dimensions)
  - Apply StandardScaler to the full combined feature matrix.

Step 8: Model Training & Evaluation
  Four models are trained and evaluated:

  Model 1 — Logistic Regression (Baseline)
    - Trained on numeric + one-hot encoded features via sklearn Pipeline.
    - Metrics: Accuracy, Macro F1, Confusion Matrix.

  Model 2 — XGBoost Classifier
    - Gradient boosting with 300 estimators, max_depth=6, lr=0.05.
    - Trained on the same preprocessed pipeline features.
    - Metrics: Accuracy, Macro F1, Confusion Matrix.

  Model 3 — FinBERT + Logistic Regression
    - Logistic Regression trained on the full fused feature matrix
      (numeric + FinBERT embeddings + FinBERT probabilities).
    - Balanced class weights to handle class imbalance.

  Model 4 — Fusion Model: FinBERT + XGBoost (Final Model)
    - Optimized XGBoost (500 estimators, max_depth=4, lr=0.03) trained on
      the full fused feature matrix with sample weighting.
    - A meta Logistic Regression stacking layer is trained on XGBoost's
      predicted class probabilities for final prediction.
    - This is the best-performing model used for inference.

Step 9: Portfolio Backtesting
  - Map predicted actions to positions: BUY=+1, HOLD=0, SELL=-1.
  - Compute strategy returns and cumulative growth curves.
  - Compare model strategy vs. a simple Always-Long (Buy & Hold) baseline.

Step 10: Single-Sample Inference
  - Demonstrate model predictions on three handcrafted test cases
    with varying market conditions (bullish, bearish, neutral).

--------------------------------------------------------------------------------
3. EXAMPLES OF PROGRAM INPUT AND OUTPUT
--------------------------------------------------------------------------------

--- INPUT FORMAT (Single Sample Prediction) ---

sample_input = {
    "asset": "TSLA",
    "news_text": "Tesla reports bankruptcy.",
    "sentiment": 0.35,
    "finbert_sentiment": 0.62,
    "ma_5": 248.50,
    "ma_10": 244.20,
    "ret_1": 0.012,
    "ret_3": 0.025,
    "ret_7": 0.041,
    "vol_5": 0.018,
    "vol_10": 0.022,
    "trend": 4.15,
    "text_len": 25,
    "sentiment_change": 0.10,
    "momentum_strength": 0.012 * 0.018,
    "return_to_vol": 0.012 / (0.018 + 1e-6),
}

--- OUTPUT (Case 1 — Bearish News, Mixed Signals) ---

  Predicted action: SELL
  Class probabilities: {
      'BUY':  0.1959,
      'HOLD': 0.2934,
      'SELL': 0.5107
  }

--- OUTPUT (Case 2 — Strong Bearish Signals) ---

  Input: "Tesla faces major losses and declining demand concerns."
         sentiment=-0.8, finbert_sentiment=-0.9, trend=-12.0

  Predicted action: BUY
  Class probabilities: {
      'BUY':  0.6702,
      'HOLD': 0.0562,
      'SELL': 0.2737
  }

--- OUTPUT (Case 3 — Strong Bullish Signals) ---

  Input: "Tesla reports record-breaking earnings and massive growth outlook."
         sentiment=0.8, finbert_sentiment=0.9, trend=15.0

  Predicted action: HOLD
  Class probabilities: {
      'BUY':  0.3238,
      'HOLD': 0.3643,
      'SELL': 0.3119
  }

Note: The model outputs a predicted action label and a probability
distribution across BUY, HOLD, and SELL for every input sample.

--------------------------------------------------------------------------------
4. SETUP AND INSTALLATION
--------------------------------------------------------------------------------

REQUIREMENTS:
  - Python 3.8 or higher
  - Google Colab (recommended, T4 GPU for FinBERT embedding generation)
    OR a local machine with a CUDA-compatible GPU

INSTALL DEPENDENCIES:
  Run the following pip commands (already included in the notebook):

    pip install transformers torch
    pip install nltk
    pip install datasets
    pip install deep-translator
    pip install xgboost scikit-learn
    pip install seaborn matplotlib pandas numpy

  Additionally, download NLTK resources inside Python:

    import nltk
    nltk.download('vader_lexicon')

--------------------------------------------------------------------------------
5. DATASET
--------------------------------------------------------------------------------

Dataset Name : TheFinAI CLEF Task 3 Trading Dataset
Source       : HuggingFace Datasets Hub
Dataset Link : https://huggingface.co/datasets/TheFinAI/CLEF_Task3_Trading

Loading the dataset in code:

    from datasets import load_dataset
    ds = load_dataset("TheFinAI/CLEF_Task3_Trading")

The dataset contains two asset splits:
  - ds['BTC']  → Bitcoin trading data with daily news and price labels
  - ds['TSLA'] → Tesla trading data with daily news and price labels

Each row contains:
  - date             : Trading date
  - asset            : Asset ticker (BTC or TSLA)
  - news             : List of news article texts for that day
  - prices           : Daily closing price
  - future_price_diff: Price difference in the next period
  - action           : Ground truth label — BUY, HOLD, or SELL
  - momentum         : Pre-computed momentum indicator
  - future_return    : Actual future return (used for backtesting)
  - 10k, 10q         : SEC filing text (mostly empty, dropped in preprocessing)

--------------------------------------------------------------------------------
6. HOW TO RUN THE SYSTEM
--------------------------------------------------------------------------------

OPTION A — Google Colab (Recommended):
  1. Open Google Colab at https://colab.research.google.com
  2. Upload the file: AIT_582_Project_3.ipynb
  3. In Runtime settings, select GPU (T4) as hardware accelerator.
  4. Run all cells from top to bottom using:
       Runtime → Run All
  5. The notebook will:
       - Install all dependencies automatically
       - Load the dataset from HuggingFace
       - Preprocess, engineer features, and train all models
       - Display evaluation metrics and visualizations
       - Run three sample inference test cases at the end

OPTION B — Local Machine:
  1. Clone or download the notebook file.
  2. Install all dependencies listed in Section 4.
  3. Launch Jupyter Notebook:
       jupyter notebook AIT_582_Project_3.ipynb
  4. Run all cells in order.
  NOTE: FinBERT embedding generation is slow without a GPU.
        Expect significantly longer runtimes on CPU-only machines.

IMPORTANT NOTES:
  - The notebook must be run sequentially from top to bottom.
    Skipping cells may cause variable reference errors.
  - The FinBERT embedding step (Cell 76) is the most time-intensive step.
    On Colab T4 GPU, expect approximately 5-15 minutes for both assets.
  - No external data files need to be downloaded manually.
    The dataset loads directly via the HuggingFace datasets library.

--------------------------------------------------------------------------------
7. MODELS USED
--------------------------------------------------------------------------------

  - ProsusAI/finbert         → Embedding extraction + classification probs
  - yiyanghkust/finbert-tone → VADER-alternative financial sentiment scoring
  - scikit-learn             → Logistic Regression, PCA, StandardScaler
  - XGBoost                  → Gradient boosting classifier (final model)
  - NLTK VADER               → Rule-based sentiment scoring

--------------------------------------------------------------------------------
8. OUTPUT FILES / ARTIFACTS
--------------------------------------------------------------------------------

  The notebook produces the following visual outputs inline:
  - Missing value heatmap
  - Price movement over time (BTC and TSLA dual-axis)
  - News sentiment distribution (overall and per asset)
  - Confusion matrices for all four models
  - Actual vs. Predicted action distribution bar charts
  - Cumulative portfolio return curves (strategy vs. buy-and-hold)
  - Feature correlation heatmap

================================================================================
END OF README
================================================================================
