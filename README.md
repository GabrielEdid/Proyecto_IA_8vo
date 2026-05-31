# SMS Classification: Ham vs Spam vs Smishing

This project contains a Jupyter notebook for classifying SMS messages into three categories:
- `ham`
- `spam`
- `smishing`

The notebook uses a standard machine learning workflow with three models:
- Complement Naive Bayes
- Logistic Regression
- Linear SVM

## Dataset

- Source: [Mendeley vmg875v4xs](https://data.mendeley.com/datasets/vmg875v4xs/1)
- File: `data/Dataset_10191.csv`
- Size: 10,191 SMS messages
- Columns: `LABEL`, `TEXT`, `URL`, `EMAIL`, `PHONE`

## Setup

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name sms-clf --display-name "Python (sms-clf)"
```

## Run

Open `notebooks/sms_classification.ipynb` in Jupyter, select the **Python (sms-clf)** kernel, and run all cells as a team.

To execute from the command line:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=sms-clf \
  notebooks/sms_classification.ipynb
```

## Notebook contents

The notebook includes:
- dataset loading
- exploratory data analysis
- text preprocessing
- feature construction with TF-IDF and simple numeric features
- training and evaluation of the three models
- confusion matrices and example predictions
