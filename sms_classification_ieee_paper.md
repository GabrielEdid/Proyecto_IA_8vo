# Multiclass Classification of SMS Messages into Ham, Spam, and Smishing Using TF-IDF and Linear Machine Learning Models

## Abstract

Short message service (SMS) remains widely used for personal communication, service notifications, marketing, and account verification. At the same time, SMS is also used for unwanted advertising and smishing attacks, where attackers attempt to deceive recipients through phishing messages delivered by text. This paper presents a supervised machine learning workflow for multiclass SMS classification into three mutually exclusive categories: ham, spam, and smishing. The study uses a consolidated dataset of 19,417 labeled SMS messages, built by reconciling three real, publicly available SMS datasets, where each message has raw text and binary indicators for URLs, email addresses, and phone numbers. The workflow includes variable understanding, exploratory data analysis, text normalization, numeric feature construction, label encoding, TF-IDF vectorization, model training, and evaluation. Three interpretable baseline models were compared: Complement Naive Bayes, Logistic Regression, and Linear Support Vector Machine (Linear SVM). Each model was trained inside a scikit-learn pipeline that combines text and numeric features while avoiding inconsistent preprocessing between training and testing. On the held-out test set, Linear SVM obtained the best performance, with 0.8862 accuracy and 0.8915 macro F1. Most remaining errors occurred between spam and smishing, while ham messages were classified with high reliability. These results show that classical linear text classification models, when combined with TF-IDF and simple structural features, provide strong and explainable performance for SMS threat classification.

**Keywords:** SMS classification, spam detection, smishing detection, TF-IDF, Logistic Regression, Complement Naive Bayes, Linear SVM, machine learning.

## I. Introduction

SMS messages are short, informal, and often contain compressed language, abbreviations, URLs, phone numbers, and service-related commands. These properties make SMS classification a useful natural language processing (NLP) problem because the model must decide whether a message is legitimate, unwanted, or malicious using limited text evidence. The three labels in this project are:

- **Ham:** a legitimate, non-malicious message.
- **Spam:** an unsolicited or promotional message.
- **Smishing:** a phishing-oriented SMS message intended to deceive the recipient into taking a risky action.

The objective of this project was to build and evaluate an understandable multiclass machine learning classifier for SMS messages. The classification task is **single-label multiclass classification**, not multilabel classification: each message belongs to exactly one of the three classes. The selected models were chosen because they are standard, explainable baselines for text classification and can be trained efficiently on sparse TF-IDF features.

The project followed five main stages: understanding the dataset variables, analyzing and preparing the data, creating machine learning models, evaluating model behavior, and interpreting the results.

## II. Dataset and Variable Understanding

The working dataset, `Dataset_consolidated.csv`, is built by reconciling three real, publicly available SMS datasets [1], [2], [3]. The three sources are summarized below:

| Source | Original classes | Reference |
|---|---|---|
| Mishra & Soni (2022), Mendeley | ham, spam, smishing | [1] |
| UCI SMS Spam Collection | ham, spam | [2] |
| Combined Smishing Dataset (Hosseinpour et al.) | spam, smishing | [3] |

Because the three files do not share the same schema or label format, a reconciliation step was applied: their columns were renamed to a common `LABEL` / `TEXT` schema, labels were normalized to lowercase and mapped to the three target classes, messages shorter than 20 characters were dropped, duplicate messages were removed (case-insensitive and whitespace-normalized, including duplicates that appear across sources), and the `URL`, `EMAIL`, and `PHONE` indicators were recomputed for every message with the same regular expressions. Finally, all available `ham` and `spam` messages were kept, while the majority class (`smishing`) was capped to the `spam` count so that no single class dominates training. The result is a consolidated file of 19,417 SMS messages with five variables:

| Variable | Type | Meaning |
|---|---:|---|
| `LABEL` | categorical | Ground-truth class: `ham`, `spam`, or `smishing`. |
| `TEXT` | text | Raw SMS message content. |
| `URL` | binary categorical | Whether the message contains a URL (`Yes`/`No`). |
| `EMAIL` | binary categorical | Whether the message contains an email address (`Yes`/`No`). |
| `PHONE` | binary categorical | Whether the message contains a phone number (`Yes`/`No`). |

The dataset is close to balanced, with 5,133 ham, 7,142 spam, and 7,142 smishing messages. Keeping the classes roughly even is useful because accuracy is less likely to hide poor performance on a minority class. The notebook also created derived variables:

- `message_length`: number of characters in the raw message.
- `word_count`: number of whitespace-separated words.
- `clean_text`: normalized text after lowercasing and replacing URLs, emails, and phone numbers with special tokens.
- `digit_count`: number of numeric characters in the message.
- `exclamation_count`: number of exclamation marks.
- `label`: numeric version of `LABEL`, generated with `LabelEncoder`.

`LabelEncoder` converts string class names into integer targets required by scikit-learn estimators [4]. In this project, the mapping was:

| Class | Encoded value |
|---|---:|
| `ham` | 0 |
| `smishing` | 1 |
| `spam` | 2 |

This encoding does not mean that `spam` is greater than `ham`; it only creates machine-readable identifiers for the target labels.

## III. Analysis and Data Preparation

### A. Exploratory Data Analysis

Exploratory data analysis (EDA) was used to understand how the classes differ before training a model. Table II summarizes the class counts.

| Class | Count |
|---|---:|
| ham | 5,133 |
| spam | 7,142 |
| smishing | 7,142 |

The dataset is close to balanced across the three labels. Message length and word count showed that ham messages were usually shorter than spam and smishing messages. Median values were:

| Class | Median characters | Median words |
|---|---:|---:|
| ham | 55 | 11 |
| spam | 132 | 22 |
| smishing | 138 | 23 |

The binary URL, email, and phone indicators also showed meaningful differences:

| Class | URL rate | Email rate | Phone rate |
|---|---:|---:|---:|
| ham | 0.001 | 0.000 | 0.002 |
| spam | 0.148 | 0.002 | 0.086 |
| smishing | 0.237 | 0.003 | 0.100 |

These results suggest that ham is easier to separate from the two suspicious classes. The more difficult distinction is between spam and smishing because both tend to be longer and may contain promotional, urgent, or action-oriented language. Among the structural indicators, the URL flag was the most discriminative: 23.7% of smishing and 14.8% of spam messages contained a URL, compared with almost none of the ham messages, while phone numbers appeared in only about 10.0% of smishing and 8.6% of spam messages.

### B. Text Cleaning

The raw `TEXT` column cannot be used directly by most classical machine learning models. The notebook therefore applied a cleaning function with the following operations:

1. Convert the message to lowercase.
2. Replace URLs with `__url__`.
3. Replace email addresses with `__email__`.
4. Replace phone numbers with `__phone__`.
5. Collapse repeated whitespace.

This approach keeps security-relevant patterns while reducing unnecessary variation. For example, two different bank-looking URLs may be replaced by the same token, allowing the model to learn that URL presence can be informative without memorizing each exact URL.

### C. Numeric Feature Construction

The notebook used both text features and simple structural features. The numeric feature set was:

- `message_length`
- `word_count`
- `digit_count`
- `exclamation_count`

These features are interpretable and relevant to the domain. Longer messages, more digits, and punctuation patterns may indicate promotional or phishing behavior. Numeric features were scaled with `MaxAbsScaler`, which is appropriate for sparse feature matrices because it scales by maximum absolute value without destroying sparsity [4].

### D. Train-Test Split

The dataset was split into 80% training data and 20% test data:

| Split | Samples | Class counts |
|---|---:|---|
| Training | 15,533 | `[4106, 5713, 5714]` |
| Test | 3,884 | `[1027, 1429, 1428]` |

The class counts follow the encoded order `[ham, smishing, spam]`. The split used `stratify=y`, meaning the class proportions were preserved in both training and test sets. This is important for fair multiclass evaluation.

## IV. Machine Learning Model Creation

### A. TF-IDF Vectorization

The main text representation was `TfidfVectorizer` [4]. TF-IDF means term frequency-inverse document frequency. It represents each message as a numeric vector where terms receive higher weight when they appear in a message but are not common across all messages. This reduces the importance of very frequent words and increases the importance of more discriminative words.

The vectorizer used:

```python
TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)
```

The configuration has three important effects:

- `stop_words='english'` removes common English words that usually carry little classification value.
- `ngram_range=(1, 2)` uses both unigrams and bigrams, so the model can learn from individual words and two-word phrases.
- `min_df=2` ignores terms that appear in only one message, reducing noise from rare tokens.

### B. ColumnTransformer

The project used `ColumnTransformer` to apply different preprocessing to different columns [4]. The `clean_text` column was transformed with TF-IDF, while the numeric columns were scaled with `MaxAbsScaler`. The transformed outputs were then concatenated into one feature matrix.

This matters because text and numeric columns require different preprocessing. Applying TF-IDF to numeric columns or numeric scaling to raw text would be incorrect.

### C. Pipeline

The preprocessing and classifier were combined with a scikit-learn `Pipeline` [4]. A pipeline stores a sequence of operations, such as:

```python
Pipeline(
    steps=[
        ('preprocessor', build_preprocessor()),
        ('classifier', classifier),
    ]
)
```

This design helps reproducibility and prevents preprocessing mistakes. During training, the pipeline learns the TF-IDF vocabulary and scaling parameters from the training set. During prediction, it applies the same learned transformations to test or new messages before classification.

### D. Complement Naive Bayes

Complement Naive Bayes is a variant of Naive Bayes designed to improve text classification behavior, especially when class distributions or word distributions create poor estimates for standard Multinomial Naive Bayes [4]. Instead of estimating class evidence only from examples inside a class, Complement Naive Bayes uses statistics from the complement of each class. In this project, it was configured as:

```python
ComplementNB(alpha=0.5)
```

The parameter `alpha` applies smoothing, which avoids zero probabilities for unseen words. This model is fast, works naturally with nonnegative sparse text features, and is often a strong baseline for text classification.

### E. Logistic Regression

Logistic Regression is a linear classifier that learns a weight for each feature and estimates class membership using a logistic or softmax decision function [4]. Although the name contains "regression," in this context it is used for classification, not numerical prediction.

The notebook used:

```python
LogisticRegression(max_iter=1000, random_state=SEED)
```

For multiclass classification, Logistic Regression learns class-specific decision scores and assigns each message to the class with the strongest score. In scikit-learn, multiclass support is built into `LogisticRegression` for compatible solvers [4]. This is different from **Linear Regression**, which predicts continuous numeric values and is not appropriate as the main model for this categorical SMS classification task.

### F. Linear Support Vector Machine

Linear SVM learns separating hyperplanes between classes by maximizing the margin between examples near the decision boundary. It is especially useful for high-dimensional text data because TF-IDF creates many sparse features. The notebook used:

```python
LinearSVC(random_state=SEED)
```

`LinearSVC` supports multiclass classification with a one-vs-rest strategy, where one classifier is trained per class and the class with the highest decision score is selected [4]. In this project, that means the model learns separate decision boundaries for ham vs. the rest, smishing vs. the rest, and spam vs. the rest.

## V. Model Evaluation and Interpretation

### A. Evaluation Metrics

The notebook evaluated the models using accuracy, precision, recall, F1-score, macro F1, classification reports, and confusion matrices. Accuracy measures the overall proportion of correct predictions. F1-score combines precision and recall, and macro F1 averages the F1-score across classes equally [4]. Macro F1 was appropriate because the project needed good behavior across all three classes, not only high total accuracy.

### B. Overall Model Comparison

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Linear SVM | 0.8862 | 0.8915 |
| Logistic Regression | 0.8834 | 0.8891 |
| Complement Naive Bayes | 0.8043 | 0.8123 |

The two linear models clearly led and were almost tied: Linear SVM achieved the highest score and was selected as the best model in the notebook, with Logistic Regression close behind. Complement Naive Bayes was noticeably weaker on this larger, more varied dataset. Because the consolidated dataset combines three different real-world sources, the messages are diverse and the spam/smishing boundary is genuinely hard, which keeps the scores realistic rather than inflated.

### C. Per-Class Classification Reports

**Complement Naive Bayes**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| ham | 0.8720 | 0.9416 | 0.9054 | 1027 |
| smishing | 0.7793 | 0.7782 | 0.7787 | 1429 |
| spam | 0.7752 | 0.7318 | 0.7529 | 1428 |

**Logistic Regression**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| ham | 0.9340 | 0.9640 | 0.9487 | 1027 |
| smishing | 0.8966 | 0.8251 | 0.8593 | 1429 |
| spam | 0.8363 | 0.8838 | 0.8594 | 1428 |

**Linear SVM**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| ham | 0.9318 | 0.9581 | 0.9448 | 1027 |
| smishing | 0.8983 | 0.8404 | 0.8684 | 1429 |
| spam | 0.8431 | 0.8803 | 0.8613 | 1428 |

The best model, Linear SVM, classified ham messages most reliably (F1 0.9448). Its lower smishing recall (0.8404) shows that some smishing messages were predicted as spam.

### D. Confusion Matrix Interpretation

For Linear SVM, the confusion matrix was:

| True class | Predicted ham | Predicted smishing | Predicted spam |
|---|---:|---:|---:|
| ham | 984 | 13 | 30 |
| smishing | 24 | 1201 | 204 |
| spam | 48 | 123 | 1257 |

The matrix shows three important patterns:

1. Ham was rarely confused with malicious classes. Of 1,027 ham messages, 984 were correct, 13 were predicted as smishing, and 30 as spam.
2. The largest single error type was smishing predicted as spam: 204 smishing messages were classified as spam.
3. Spam and smishing overlap more than ham and the malicious classes (123 spam messages were also predicted as smishing). This matches the EDA findings because both spam and smishing are longer, may contain URLs or phone numbers, and often use persuasive or urgent language.

### E. Practical Interpretation

The results suggest that classical ML models remain effective for this dataset. The strong Linear SVM result is expected for sparse text classification because linear margin-based models work well when each document is represented by many TF-IDF features. Logistic Regression also performed strongly and has the advantage of probabilistic interpretation. Complement Naive Bayes was weaker on this consolidated dataset but remains useful as a simple and fast baseline.

The model does not fully solve the semantic difference between spam and smishing. Spam and smishing can share phrases such as "congratulations," "offer," "account," "call now," or "verify." A future model could include more security-specific features, such as URL domain reputation, sender metadata, named entities, or contextual embeddings from transformer models.

## VI. Limitations

This project has several limitations. First, the dataset was made close to balanced by capping the majority class; real-world SMS traffic is usually imbalanced, with ham messages appearing much more frequently than spam or smishing. Second, the dataset contains explicit URL, email, and phone indicators that may not always be available in production unless extracted reliably. Third, the model uses surface-level text and structural features, not deeper semantic or threat-intelligence features. Fourth, the notebook used a single train-test split; cross-validation could provide a more robust estimate of model stability. Finally, although the three source datasets are real and publicly available, they were collected through different processes and time periods, so real-world generalization should still be tested with independently collected live SMS data.

## VII. Conclusion

This paper described an end-to-end SMS multiclass classification workflow for ham, spam, and smishing detection. The process began by reconciling three real public datasets and understanding the dataset variables, then used exploratory analysis to identify class differences in length, words, URLs, emails, and phone numbers. The modeling pipeline cleaned the text, encoded labels, built numeric features, converted text into TF-IDF vectors, and trained three interpretable classifiers.

The best-performing model was Linear SVM, with 0.8862 accuracy and 0.8915 macro F1 on the held-out test set. The evaluation showed that ham messages were the easiest to classify, while spam and smishing were the most commonly confused classes. Overall, the project demonstrates that a well-prepared TF-IDF pipeline with classical linear models can provide strong, explainable SMS classification performance on a realistic, multi-source dataset.

## References

[1] S. Mishra and D. Soni, "SMS Phishing Dataset for Machine Learning and Pattern Recognition," Mendeley Data, 2022. [Online]. Available: https://data.mendeley.com/datasets/f45bkkt8pr/1

[2] T. A. Almeida, J. M. G. Hidalgo, and A. Yamakami, "SMS Spam Collection," UCI Machine Learning Repository. [Online]. Available: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

[3] S. Hosseinpour *et al.*, "Combined Smishing Dataset," GitHub repository. [Online]. Available: https://github.com/shaghayegh-hp/Smishing_Dataset

[4] scikit-learn developers, "User Guide," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/user_guide.html
