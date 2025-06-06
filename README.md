# Sentiment Analysis on Twitter Data

This repository contains a comprehensive study on sentiment analysis of Twitter data using both traditional machine learning and deep learning models. The project explores various approaches, including classical models (SVM, TextBlob, Vader) and advanced neural networks (LSTM, Bi-LSTM, GRU, CNN, FNN, BERT, RoBERTa).

## Project Structure

```
Sentiment_Analyses_Twitter/
├── data/
│   ├── twitter_training.csv
│   └── twitter_validation.csv
├── deep_learning/
│   ├── BERT.ipynb
│   ├── bi-lstm.ipynb
│   ├── cnn.ipynb
│   ├── gru.ipynb
│   ├── lstm.ipynb
│   └── ...
├── machine_learning/
│   └── svm.ipynb
├── traditional_methods/
│   ├── textblob.ipynb
│   └── vader.ipynb
├── utils/
│   ├── data_processing.py
│   └── data_visualization.ipynb
├── latex_files/ (paper and images)
└── README.md
```

## Data
- `data/twitter_training.csv` and `data/twitter_validation.csv` contain labeled tweets with columns: `tweetID`, `entity`, `sentiment`, `tweet_content`.
- Sentiment labels: `positive`, `negative`, `neutral`, and some `irrelevant` (filtered in code).

## Approaches
- **Traditional Methods:**
  - SVM (with BoW/TF-IDF)
  - TextBlob
  - Vader
- **Deep Learning:**
  - Feedforward Neural Network (FNN)
  - LSTM, Bi-LSTM, GRU
  - CNN
  - Transformers: BERT, RoBERTa

## Notebooks
- Each notebook in `deep_learning/`, `machine_learning/`, and `traditional_methods/` demonstrates a different model or approach.
- Data preprocessing and cleaning are handled in each notebook, with reusable functions in `utils/data_processing.py`.

## Reproducibility
- All random seeds are set for reproducibility.
- Data cleaning and preprocessing steps are consistent across notebooks.

## Paper
- The `latex_files/` directory contains the LaTeX source for the accompanying paper, including figures and results.

## Acknowledgements
- Twitter data is for research purposes only.
- Built with open-source libraries: scikit-learn, PyTorch, HuggingFace Transformers, NLTK, etc.

