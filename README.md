# 🛡️ AI Spam Message Detector

A portfolio-ready Natural Language Processing project that classifies SMS/email messages as **spam** or **legitimate (ham)**.

## What this project demonstrates

- Real public dataset
- Text cleaning and deduplication
- TF-IDF feature engineering
- Model comparison
- Naive Bayes
- Logistic Regression
- Linear SVM
- Accuracy, precision, recall and F1
- Confusion matrix
- Interactive probability visualization
- Streamlit web application

## Dataset

The project uses the **UCI SMS Spam Collection**, containing 5,574 labeled English SMS messages. The dataset is licensed under **CC BY 4.0**.

Source:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Setup

```bash
pip install -r requirements.txt
```

Download the real dataset:

```bash
python src/download_data.py
```

Train and compare the models:

```bash
python src/train_model.py
```

Run the application:

```bash
streamlit run app.py
```

## Important

Do not claim the test metrics represent all real-world messages. The model is trained on the UCI dataset and may perform differently on modern, multilingual, or region-specific spam.

## Portfolio files

After training, these are generated:

- `models/spam_detector.pkl`
- `reports/model_comparison.csv`
- `reports/metrics.json`
- `reports/confusion_matrix.png`
- individual classification reports in `reports/`

Streamlit deployed page link
https://brilliant-ai-tech-spam-message-detector-app-tbcxk9.streamlit.app/