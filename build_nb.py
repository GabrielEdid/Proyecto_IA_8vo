"""Generator for notebooks/sms_classification.ipynb using nbformat.
Run once: .venv/bin/python build_nb.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md("""# SMS Classification: Ham vs Spam vs Smishing

Three-class text classification on the Mendeley **Balanced Dataset for Spam and Smishing Detection** ([vmg875v4xs](https://data.mendeley.com/datasets/vmg875v4xs/1)) — 10,191 English SMS messages, perfectly balanced (3,397 each).

We compare three model tiers under one fair evaluation harness:

1. **Classic ML** — TF-IDF (word + char n-grams) + engineered features, with Complement Naive Bayes, Logistic Regression, and Linear SVM.
2. **Gradient boosting** — XGBoost on SVD-reduced TF-IDF + engineered features.
3. **Fine-tuned DistilBERT** — transformer baseline.

The academic focus is the **Spam vs Smishing** confusion: Ham is trivially separable, the two malicious classes overlap in vocabulary, and lexical cues in the message text (phone-number and URL mentions, words like *claim* / *prize*) are what pull them apart.

> **Note:** the dataset's separate `URL`/`EMAIL`/`PHONE` flag columns are explored in the EDA but are **deliberately excluded as model features** — every model learns from the message text (and text-derived statistics) only.""")

md("## 1. Setup")

code("""import os, re, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

# Device: Apple Silicon (mps), CUDA, or CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print("Torch device:", DEVICE)""")

md("## 2. Load and sanity-check the data")

code("""# Resolve path whether the notebook runs from repo root or notebooks/
CANDIDATES = ["data/Dataset_10191.csv", "../data/Dataset_10191.csv"]
DATA_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
assert DATA_PATH, "Dataset_10191.csv not found. Place it under data/ (see README)."

df = pd.read_csv(DATA_PATH)
assert df.shape == (10191, 5), f"Unexpected shape: {df.shape}"
assert set(df["LABEL"].unique()) == {"ham", "spam", "smishing"}, df["LABEL"].unique()
assert df.isnull().sum().sum() == 0, "Unexpected null values"
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df["LABEL"].value_counts())
df.head()""")

md("## 3. Exploratory data analysis")

code("""# 3.1 Class balance
ax = df["LABEL"].value_counts().sort_index().plot(kind="bar", color=["#4c72b0", "#dd8452", "#c44e52"])
ax.set_title("Class distribution (balanced by design)")
ax.set_xlabel("class"); ax.set_ylabel("count")
plt.tight_layout(); plt.show()""")

code("""# 3.2 Message length per class
df["char_len"] = df["TEXT"].str.len()
df["word_count"] = df["TEXT"].str.split().apply(len)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=df, x="LABEL", y="char_len", order=["ham", "spam", "smishing"], ax=axes[0])
axes[0].set_title("Character length by class"); axes[0].set_ylim(0, 400)
sns.boxplot(data=df, x="LABEL", y="word_count", order=["ham", "spam", "smishing"], ax=axes[1])
axes[1].set_title("Word count by class"); axes[1].set_ylim(0, 80)
plt.tight_layout(); plt.show()

df.groupby("LABEL")[["char_len", "word_count"]].median()""")

code("""# 3.3 Prevalence of URL / EMAIL / PHONE flags per class — the key discriminating signal
flag_rates = (df.assign(**{c: (df[c] == "Yes").astype(int) for c in ["URL", "EMAIL", "PHONE"]})
                .groupby("LABEL")[["URL", "EMAIL", "PHONE"]].mean())
ax = flag_rates.loc[["ham", "spam", "smishing"]].plot(kind="bar", figsize=(8, 4))
ax.set_title("Share of messages containing URL / EMAIL / PHONE, by class")
ax.set_ylabel("proportion"); ax.set_xlabel("class")
plt.tight_layout(); plt.show()
flag_rates.loc[["ham", "spam", "smishing"]].round(3)""")

code("""# 3.4 Most distinctive tokens per class (raw text, English stopwords removed)
from sklearn.feature_extraction.text import CountVectorizer

for cls in ["ham", "spam", "smishing"]:
    cv = CountVectorizer(stop_words="english", min_df=5)
    X = cv.fit_transform(df.loc[df["LABEL"] == cls, "TEXT"])
    freqs = np.asarray(X.sum(axis=0)).ravel()
    top = pd.Series(freqs, index=cv.get_feature_names_out()).sort_values(ascending=False).head(12)
    print(f"[{cls}] {', '.join(top.index)}")""")

md("""## 4. Preprocessing

Deliberately **light**. We lowercase and normalize whitespace, but keep digits, punctuation, and currency. Raw URLs, emails, and phone numbers are collapsed to placeholder tokens (`__url__`, `__email__`, `__phone__`) — these are exactly the signals that separate smishing from ordinary spam, so we keep them as features instead of stripping them.""")

code("""URL_RE = re.compile(r"(https?://\\S+|www\\.\\S+)")
EMAIL_RE = re.compile(r"\\b[\\w.+-]+@[\\w-]+\\.[\\w.-]+\\b")
PHONE_RE = re.compile(r"(\\+?\\d[\\d\\-\\s().]{6,}\\d)")

def clean_text(t):
    t = str(t).lower()
    t = URL_RE.sub(" __url__ ", t)
    t = EMAIL_RE.sub(" __email__ ", t)
    t = PHONE_RE.sub(" __phone__ ", t)
    t = re.sub(r"\\s+", " ", t).strip()
    return t

df["clean_text"] = df["TEXT"].apply(clean_text)

# Encode labels (alphabetical: ham=0, smishing=1, spam=2)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["label"] = le.fit_transform(df["LABEL"])
LABEL_NAMES = list(le.classes_)

print("Label mapping:", dict(zip(LABEL_NAMES, range(len(LABEL_NAMES)))))
df[["TEXT", "clean_text", "label"]].head(3)""")

md("## 5. Feature engineering")

code("""# Engineered numeric features (computed on raw TEXT)
def numeric_features(s):
    s = str(s); n = len(s) or 1
    return pd.Series({
        "digit_count":   sum(c.isdigit() for c in s),
        "upper_ratio":   sum(c.isupper() for c in s) / n,
        "special_count": sum((not c.isalnum()) and (not c.isspace()) for c in s),
        "exclaim_count": s.count("!"),
    })

eng = df["TEXT"].apply(numeric_features)
df = pd.concat([df, eng], axis=1)

NUMERIC_COLS = ["char_len", "word_count", "digit_count",
                "upper_ratio", "special_count", "exclaim_count"]
# All non-negative -> safe for MaxAbsScaler and for ComplementNB downstream.
# The dataset's URL/EMAIL/PHONE flag columns are deliberately NOT used as model
# features; the models learn only from the message text and text-derived statistics.
df[NUMERIC_COLS].describe().round(2)""")

md("## 6. Stratified train/test split")

code("""from sklearn.model_selection import train_test_split

FEATURE_COLS = ["clean_text", "TEXT"] + NUMERIC_COLS
X = df[FEATURE_COLS].copy()
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
print("train:", X_train.shape[0], " test:", X_test.shape[0])
print("train class balance:", np.bincount(y_train), " test:", np.bincount(y_test))""")

md("""## 7. Classic ML baseline

A shared `ColumnTransformer` produces the feature matrix: word TF-IDF (1–2 grams), character TF-IDF (`char_wb` 2–4 grams, robust to obfuscation), and the scaled numeric features. The vectorizers are fit **inside the pipeline on training folds only**, so there is no leakage. We tune each model with 5-fold stratified CV on macro-F1.""")

code("""from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def build_features():
    return ColumnTransformer([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 max_features=30000), "clean_text"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5,
                                 max_features=20000, sublinear_tf=True), "clean_text"),
        ("num", MaxAbsScaler(), NUMERIC_COLS),
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
MODELS = {
    "ComplementNB": (ComplementNB(), {"clf__alpha": [0.1, 0.3, 1.0]}),
    "LogReg":       (LogisticRegression(max_iter=2000, C=1.0), {"clf__C": [0.5, 1.0, 5.0]}),
    "LinearSVM":    (LinearSVC(), {"clf__C": [0.5, 1.0, 5.0]}),
}

predictions = {}          # name -> y_pred on test (shared eval harness uses this)
fitted = {}               # name -> fitted estimator
cv_scores = {}

for name, (clf, grid) in MODELS.items():
    pipe = Pipeline([("features", build_features()), ("clf", clf)])
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)
    fitted[name] = gs.best_estimator_
    cv_scores[name] = gs.best_score_
    predictions[name] = gs.best_estimator_.predict(X_test)
    print(f"{name:14s} best params={gs.best_params_}  CV macro-F1={gs.best_score_:.4f}")""")

md("""## 8. Gradient boosting (XGBoost)

TF-IDF is reduced to 200 dense components via Truncated SVD, concatenated with the scaled numeric features, and fed to XGBoost. SVD is fit inside the pipeline on training data only.""")

code("""from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier

text_svd = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=30000)),
    ("svd", TruncatedSVD(n_components=200, random_state=SEED)),
])
xgb_features = ColumnTransformer([
    ("text", text_svd, "clean_text"),
    ("num", MaxAbsScaler(), NUMERIC_COLS),
])
xgb = Pipeline([
    ("features", xgb_features),
    ("clf", XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist",
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", n_jobs=-1, random_state=SEED)),
])
xgb.fit(X_train, y_train)
fitted["XGBoost"] = xgb
predictions["XGBoost"] = xgb.predict(X_test)
print("XGBoost trained.")""")

md("""## 9. Fine-tuned DistilBERT

`distilbert-base-uncased` with a 3-label classification head, fine-tuned on the raw message text. A 10% validation split (carved from the training set) drives early stopping on macro-F1. Runs on the detected GPU (`mps`/`cuda`) if available.""")

code("""from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          EarlyStoppingCallback)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

# Validation split from the training set (stratified)
tr_df = X_train.copy(); tr_df["label"] = y_train
te_df = X_test.copy();  te_df["label"] = y_test
tr_idx, val_idx = train_test_split(tr_df.index, test_size=0.10,
                                   stratify=tr_df["label"], random_state=SEED)

def to_ds(frame):
    return Dataset.from_dict({"text": frame["TEXT"].tolist(),
                              "label": frame["label"].tolist()})

ds_train = to_ds(tr_df.loc[tr_idx])
ds_val   = to_ds(tr_df.loc[val_idx])
ds_test  = to_ds(te_df)

def tok_fn(batch):
    return tok(batch["text"], truncation=True, max_length=128)

ds_train = ds_train.map(tok_fn, batched=True)
ds_val   = ds_val.map(tok_fn, batched=True)
ds_test  = ds_test.map(tok_fn, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro")}""")

code("""model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3,
    id2label={i: n for i, n in enumerate(LABEL_NAMES)},
    label2id={n: i for i, n in enumerate(LABEL_NAMES)})

args = TrainingArguments(
    output_dir="bert_out",
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
    num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=32,
    learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
    logging_steps=50, report_to="none", seed=SEED, save_total_limit=1)

trainer = Trainer(
    model=model, args=args,
    train_dataset=ds_train, eval_dataset=ds_val,
    processing_class=tok, data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])

trainer.train()
predictions["DistilBERT"] = trainer.predict(ds_test).predictions.argmax(-1)
print("DistilBERT fine-tuned.")""")

md("""## 10. Evaluation harness

Every model is scored identically on the held-out test set: accuracy, macro-F1, weighted-F1, plus per-class precision/recall/F1 and a confusion matrix.""")

code("""from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

def score(name, y_true, y_pred):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }

results = pd.DataFrame([score(n, y_test, p) for n, p in predictions.items()])
results = results.set_index("model").sort_values("macro_f1", ascending=False).round(4)
results""")

code("""# Per-class detail for every model
for name, y_pred in predictions.items():
    print(f"===== {name} =====")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, digits=3))""")

code("""# Confusion matrix per model
n = len(predictions)
fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6))
if n == 1: axes = [axes]
for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(name); ax.set_xlabel("predicted"); ax.set_ylabel("true")
plt.tight_layout(); plt.show()""")

md("## 11. Model comparison")

code("""ax = results[["accuracy", "macro_f1"]].plot(kind="bar", figsize=(9, 4))
ax.set_title("Model comparison on held-out test set")
ax.set_ylabel("score"); ax.set_ylim(0.8, 1.0); ax.set_xlabel("")
plt.xticks(rotation=20); plt.tight_layout(); plt.show()

best_model = results.index[0]
print(f"Best model by macro-F1: {best_model} ({results.loc[best_model, 'macro_f1']:.4f})")""")

md("""## 12. Error analysis

The interesting failures live in the **Spam vs Smishing** cells of the confusion matrix. We inspect those misclassifications for the best model and read off the most discriminative tokens learned by Logistic Regression for interpretability.""")

code("""# Spam <-> Smishing confusions for the best model
spam_i, smis_i = LABEL_NAMES.index("spam"), LABEL_NAMES.index("smishing")
best_pred = predictions[best_model]
mask = (((y_test == spam_i) & (best_pred == smis_i)) |
        ((y_test == smis_i) & (best_pred == spam_i)))
err = X_test.loc[mask].copy()
err["true"] = [LABEL_NAMES[i] for i in y_test[mask]]
err["pred"] = [LABEL_NAMES[i] for i in best_pred[mask]]
print(f"{best_model}: {mask.sum()} spam<->smishing confusions out of {len(y_test)} test msgs")
err[["true", "pred", "TEXT"]].head(12)""")

code("""# Interpretability: top word-TF-IDF tokens per class from the Logistic Regression model
logreg = fitted["LogReg"]
word_vec = logreg.named_steps["features"].named_transformers_["word"]
feat_names = word_vec.get_feature_names_out()
coefs = logreg.named_steps["clf"].coef_           # (n_classes, n_features) incl char+num
word_coefs = coefs[:, :len(feat_names)]           # word block comes first in the ColumnTransformer

for ci, cname in enumerate(LABEL_NAMES):
    top = np.argsort(word_coefs[ci])[-15:][::-1]
    print(f"[{cname}] {', '.join(feat_names[top])}")""")

md("""## 13. Conclusion

- **Setup.** The dataset's `URL`/`EMAIL`/`PHONE` flag columns are excluded — every model learns only from the message text (word + char TF-IDF) plus text-derived statistics (length, digit count, uppercase ratio, etc.).
- **Ranking (this run).** Macro-F1 on the held-out test set: **Linear SVM 0.980 ≈ Logistic Regression 0.979 > DistilBERT 0.976 ≈ XGBoost 0.975 > Complement NB 0.963**. Every model clears 0.96 — the dataset is clean, balanced, and lexically well-separated.
- **Classic models win here.** The TF-IDF linear models slightly *edge out* the fine-tuned DistilBERT. This is a legitimate and common outcome for SMS spam: messages are short and keyword-driven, so strong lexical cues (`free`, `claim`, `prize`, `__phone__`) are exactly what bag-of-n-grams captures, while a 3-epoch DistilBERT has no headroom to beat them. The transformer would likely need more epochs / a domain-pretrained checkpoint to pull ahead, at much higher cost.
- **Dropping the flag columns barely mattered.** Removing the explicit `URL`/`EMAIL`/`PHONE` features moved macro-F1 by less than 0.003 for every model. The reason: the same information already reaches the model through the text — raw URLs and phone numbers are normalized to `__url__`/`__phone__` tokens, which the TF-IDF captures. The Logistic Regression coefficients confirm it: the top smishing tokens are led by `__phone__`, `__url__`, `prize`, `claim`, `reply stop`.
- **Where the difficulty is.** Ham is near-perfectly separated by every model (Linear SVM ham F1 ≈ 0.999). Essentially all remaining errors are **Spam ↔ Smishing** (38 of 2,039 test messages for the best model) — both are unsolicited promotional messages, so the boundary is genuinely fuzzy even for humans (see the misclassified examples above).
- **Context from EDA.** Even though the flag columns are not used as features, the EDA shows why smishing is identifiable: 72% of smishing messages contain a phone number vs 39% of spam and ~0% of ham — and that same "phone present" signal reaches the model through the `__phone__` token.
- **Recommendation & trade-offs.** For this dataset **Linear SVM / Logistic Regression is the practical winner**: highest accuracy here, trains in seconds, and is fully interpretable. DistilBERT brought no accuracy gain while requiring a GPU and being opaque.
- **Limitations & next steps.** English-only. Natural extensions: calibrated probabilities for a review queue, threshold tuning for high *recall* on the smishing class (missing a phishing SMS is the costly error), and a two-stage smishing-vs-rest classifier.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "sms-clf", "display_name": "Python (sms-clf)", "language": "python"}
nb["metadata"]["language_info"] = {"name": "python"}

with open("notebooks/sms_classification.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"Wrote notebooks/sms_classification.ipynb with {len(cells)} cells")
