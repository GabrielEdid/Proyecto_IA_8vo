# SMS Classification: Ham vs Spam vs Smishing

Three-class text classification on the Mendeley **Balanced Dataset for Spam and Smishing Detection**, comparing classic ML, gradient boosting, and a fine-tuned transformer.

## Dataset

- Source: [Mendeley vmg875v4xs](https://data.mendeley.com/datasets/vmg875v4xs/1) (CC BY 4.0)
- `Dataset_10191.csv`: 10,191 English SMS messages, perfectly balanced (3,397 each of `ham`, `spam`, `smishing`)
- Columns: `LABEL`, `TEXT`, `URL`, `EMAIL`, `PHONE` (last three are Yes/No flags derived by regex)

The CSV is already in `data/`. To re-download:

```bash
curl -sL "https://data.mendeley.com/public-files/datasets/vmg875v4xs/files/f167b0a7-c411-45d4-9cbc-ee06c3b42753/file_downloaded" -o data/Dataset_10191.csv
```

## Setup

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name sms-clf --display-name "Python (sms-clf)"
```

## Run

Open `notebooks/sms_classification.ipynb` and select the **Python (sms-clf)** kernel, then Run All. Or headless:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=sms-clf \
  notebooks/sms_classification.ipynb
```

DistilBERT fine-tuning uses the GPU automatically (Apple Silicon `mps` or CUDA) and falls back to CPU.

## Approach

| Tier | Models | Features |
|------|--------|----------|
| Classic ML | Complement NB, Logistic Regression, Linear SVM | word + char TF-IDF + engineered numeric |
| Gradient boosting | XGBoost | SVD-reduced TF-IDF + engineered numeric |
| Transformer | fine-tuned DistilBERT | raw text |

All models share one evaluation harness (stratified 80/20 split, accuracy + macro-F1 + per-class metrics + confusion matrices). Preprocessing is intentionally light — URLs/emails/phones become placeholder tokens rather than being stripped, since those are the strongest smishing signals. Ham is trivially separable; the modeling challenge is the **Spam vs Smishing** boundary.
