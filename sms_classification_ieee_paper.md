# Multiclass Classification of SMS Messages into Ham, Spam, and Smishing Using TF-IDF and Linear Machine Learning Models

## Abstract

Short message service (SMS) remains widely used for personal communication, service notifications, marketing, and account verification. At the same time, SMS is also used for unwanted advertising and smishing attacks, where attackers attempt to deceive recipients through phishing messages delivered by text. This paper presents a supervised machine learning workflow for multiclass SMS classification into three mutually exclusive categories: ham, spam, and smishing. The study uses a balanced dataset of 10,191 labeled SMS messages with raw text and binary indicators for URLs, email addresses, and phone numbers. The workflow includes variable understanding, exploratory data analysis, text normalization, numeric feature construction, label encoding, TF-IDF vectorization, model training, and evaluation. Three interpretable baseline models were compared: Complement Naive Bayes, Logistic Regression, and Linear Support Vector Machine (Linear SVM). Each model was trained inside a scikit-learn pipeline that combines text and numeric features while avoiding inconsistent preprocessing between training and testing. On the held-out test set, Linear SVM obtained the best performance, with 0.9789 accuracy and 0.9789 macro F1. Most remaining errors occurred between spam and smishing, while ham messages were classified with very high reliability. These results show that classical linear text classification models, when combined with TF-IDF and simple structural features, provide strong and explainable performance for SMS threat classification.

**Keywords:** SMS classification, spam detection, smishing detection, TF-IDF, Logistic Regression, Complement Naive Bayes, Linear SVM, machine learning.

## I. Introduction

SMS messages are short, informal, and often contain compressed language, abbreviations, URLs, phone numbers, and service-related commands. These properties make SMS classification a useful natural language processing (NLP) problem because the model must decide whether a message is legitimate, unwanted, or malicious using limited text evidence. The three labels in this project are:

- **Ham:** a legitimate, non-malicious message.
- **Spam:** an unsolicited or promotional message.
- **Smishing:** a phishing-oriented SMS message intended to deceive the recipient into taking a risky action.

The objective of this project was to build and evaluate an understandable multiclass machine learning classifier for SMS messages. The classification task is **single-label multiclass classification**, not multilabel classification: each message belongs to exactly one of the three classes. The selected models were chosen because they are standard, explainable baselines for text classification and can be trained efficiently on sparse TF-IDF features.

The project followed five main stages: understanding the dataset variables, analyzing and preparing the data, creating machine learning models, evaluating model behavior, and interpreting the results.

## II. Dataset and Variable Understanding

The dataset used in the notebook is `Dataset_10191.csv`, obtained from Mendeley Data [1]. It contains 10,191 SMS messages and five original variables:

| Variable | Type | Meaning |
|---|---:|---|
| `LABEL` | categorical | Ground-truth class: `ham`, `spam`, or `smishing`. |
| `TEXT` | text | Raw SMS message content. |
| `URL` | binary categorical | Whether the message contains a URL (`Yes`/`No`). |
| `EMAIL` | binary categorical | Whether the message contains an email address (`Yes`/`No`). |
| `PHONE` | binary categorical | Whether the message contains a phone number (`Yes`/`No`). |

The dataset is balanced, with 3,397 examples per class. This balance is useful because accuracy is less likely to hide poor performance on a minority class. The notebook also created derived variables:

- `message_length`: number of characters in the raw message.
- `word_count`: number of whitespace-separated words.
- `clean_text`: normalized text after lowercasing and replacing URLs, emails, and phone numbers with special tokens.
- `digit_count`: number of numeric characters in the message.
- `exclamation_count`: number of exclamation marks.
- `label`: numeric version of `LABEL`, generated with `LabelEncoder`.

`LabelEncoder` converts string class names into integer targets required by scikit-learn estimators [2]. In this project, the mapping was:

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
| ham | 3,397 |
| spam | 3,397 |
| smishing | 3,397 |

The dataset is perfectly balanced across the three labels. Message length and word count showed that ham messages were usually shorter than spam and smishing messages. Median values were:

| Class | Median characters | Median words |
|---|---:|---:|
| ham | 52 | 11 |
| spam | 145 | 24 |
| smishing | 147 | 24 |

The binary URL, email, and phone indicators also showed meaningful differences:

| Class | URL rate | Email rate | Phone rate |
|---|---:|---:|---:|
| ham | 0.000 | 0.002 | 0.001 |
| spam | 0.046 | 0.032 | 0.390 |
| smishing | 0.089 | 0.022 | 0.721 |

These results suggest that ham is easier to separate from the two suspicious classes. The more difficult distinction is between spam and smishing because both tend to be longer and may contain promotional, urgent, or action-oriented language. Smishing had the strongest phone-number signal, with 72.1% of messages containing a phone number.

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

These features are interpretable and relevant to the domain. Longer messages, more digits, and punctuation patterns may indicate promotional or phishing behavior. Numeric features were scaled with `MaxAbsScaler`, which is appropriate for sparse feature matrices because it scales by maximum absolute value without destroying sparsity [3].

### D. Train-Test Split

The dataset was split into 80% training data and 20% test data:

| Split | Samples | Class counts |
|---|---:|---|
| Training | 8,152 | `[2718, 2717, 2717]` |
| Test | 2,039 | `[679, 680, 680]` |

The split used `stratify=y`, meaning the class proportions were preserved in both training and test sets. This is important for fair multiclass evaluation.

## IV. Machine Learning Model Creation

### A. TF-IDF Vectorization

The main text representation was `TfidfVectorizer` [4]. TF-IDF means term frequency-inverse document frequency. It represents each message as a numeric vector where terms receive higher weight when they appear in a message but are not common across all messages. This reduces the importance of very frequent words and increases the importance of more discriminative words [5].

The vectorizer used:

```python
TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)
```

The configuration has three important effects:

- `stop_words='english'` removes common English words that usually carry little classification value.
- `ngram_range=(1, 2)` uses both unigrams and bigrams, so the model can learn from individual words and two-word phrases.
- `min_df=2` ignores terms that appear in only one message, reducing noise from rare tokens.

### B. ColumnTransformer

The project used `ColumnTransformer` to apply different preprocessing to different columns [6]. The `clean_text` column was transformed with TF-IDF, while the numeric columns were scaled with `MaxAbsScaler`. The transformed outputs were then concatenated into one feature matrix.

This matters because text and numeric columns require different preprocessing. Applying TF-IDF to numeric columns or numeric scaling to raw text would be incorrect.

### C. Pipeline

The preprocessing and classifier were combined with a scikit-learn `Pipeline` [6]. A pipeline stores a sequence of operations, such as:

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

Complement Naive Bayes is a variant of Naive Bayes designed to improve text classification behavior, especially when class distributions or word distributions create poor estimates for standard Multinomial Naive Bayes [7], [8]. Instead of estimating class evidence only from examples inside a class, Complement Naive Bayes uses statistics from the complement of each class. In this project, it was configured as:

```python
ComplementNB(alpha=0.5)
```

The parameter `alpha` applies smoothing, which avoids zero probabilities for unseen words. This model is fast, works naturally with nonnegative sparse text features, and is often a strong baseline for text classification.

### E. Logistic Regression

Logistic Regression is a linear classifier that learns a weight for each feature and estimates class membership using a logistic or softmax decision function [9]. Although the name contains "regression," in this context it is used for classification, not numerical prediction.

The notebook used:

```python
LogisticRegression(max_iter=1000, random_state=SEED)
```

For multiclass classification, Logistic Regression learns class-specific decision scores and assigns each message to the class with the strongest score. In scikit-learn, multiclass support is built into `LogisticRegression` for compatible solvers [9], [10]. This is different from **Linear Regression**, which predicts continuous numeric values and is not appropriate as the main model for this categorical SMS classification task.

### F. Linear Support Vector Machine

Linear SVM learns separating hyperplanes between classes by maximizing the margin between examples near the decision boundary [11], [12]. It is especially useful for high-dimensional text data because TF-IDF creates many sparse features. The notebook used:

```python
LinearSVC(random_state=SEED)
```

`LinearSVC` supports multiclass classification with a one-vs-rest strategy, where one classifier is trained per class and the class with the highest decision score is selected [10], [13]. In this project, that means the model learns separate decision boundaries for ham vs. the rest, smishing vs. the rest, and spam vs. the rest.

## V. Model Evaluation and Interpretation

### A. Evaluation Metrics

The notebook evaluated the models using accuracy, precision, recall, F1-score, macro F1, classification reports, and confusion matrices. Accuracy measures the overall proportion of correct predictions. F1-score combines precision and recall, and macro F1 averages the F1-score across classes equally [14]. Macro F1 was appropriate because the project needed good behavior across all three classes, not only high total accuracy.

### B. Overall Model Comparison

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Linear SVM | 0.9789 | 0.9789 |
| Logistic Regression | 0.9681 | 0.9682 |
| Complement Naive Bayes | 0.9578 | 0.9580 |

All three models performed well, which indicates that the selected TF-IDF and structural features captured important class patterns. Linear SVM achieved the highest score and was selected as the best model in the notebook.

### C. Per-Class Classification Reports

**Complement Naive Bayes**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| ham | 0.9970 | 0.9661 | 0.9813 | 679 |
| smishing | 0.9289 | 0.9603 | 0.9443 | 680 |
| spam | 0.9499 | 0.9471 | 0.9485 | 680 |

**Logistic Regression**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| ham | 0.9911 | 0.9867 | 0.9889 | 679 |
| smishing | 0.9714 | 0.9485 | 0.9598 | 680 |
| spam | 0.9428 | 0.9691 | 0.9558 | 680 |

**Linear SVM**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| ham | 1.0000 | 0.9971 | 0.9985 | 679 |
| smishing | 0.9818 | 0.9544 | 0.9679 | 680 |
| spam | 0.9558 | 0.9853 | 0.9703 | 680 |

The best model, Linear SVM, classified ham messages almost perfectly. It also performed strongly on spam and smishing, but its lower smishing recall shows that some smishing messages were predicted as spam.

### D. Confusion Matrix Interpretation

For Linear SVM, the confusion matrix was:

| True class | Predicted ham | Predicted smishing | Predicted spam |
|---|---:|---:|---:|
| ham | 677 | 2 | 0 |
| smishing | 0 | 649 | 31 |
| spam | 0 | 10 | 670 |

The matrix shows three important patterns:

1. Ham was rarely confused with malicious classes. Only 2 ham messages were predicted as smishing, and none were predicted as spam.
2. The main error type was smishing predicted as spam: 31 smishing messages were classified as spam.
3. Spam and smishing overlap more than ham and malicious messages. This matches the EDA findings because both spam and smishing are longer, may contain URLs or phone numbers, and often use persuasive or urgent language.

### E. Practical Interpretation

The results suggest that classical ML models remain highly effective for this dataset. The strong Linear SVM result is expected for sparse text classification because linear margin-based models work well when each document is represented by many TF-IDF features [12]. Logistic Regression also performed strongly and has the advantage of probabilistic interpretation. Complement Naive Bayes was slightly weaker but remains useful as a simple and fast baseline.

The model does not fully solve the semantic difference between spam and smishing. Spam and smishing can share phrases such as "congratulations," "offer," "account," "call now," or "verify." A future model could include more security-specific features, such as URL domain reputation, sender metadata, named entities, or contextual embeddings from transformer models.

## VI. Limitations

This project has several limitations. First, the dataset is balanced, which is helpful for model training, but real-world SMS traffic is usually imbalanced, with ham messages appearing much more frequently than spam or smishing. Second, the dataset contains explicit URL, email, and phone indicators that may not always be available in production unless extracted reliably. Third, the model uses surface-level text and structural features, not deeper semantic or threat-intelligence features. Fourth, the notebook used a single train-test split; cross-validation could provide a more robust estimate of model stability. Finally, the dataset was generated or balanced using an LLM-assisted process according to its source description [1], so real-world generalization should be tested with independent live or naturally collected SMS data.

## VII. Conclusion

This paper described an end-to-end SMS multiclass classification workflow for ham, spam, and smishing detection. The process began by understanding the dataset variables, then used exploratory analysis to identify class differences in length, words, URLs, emails, and phone numbers. The modeling pipeline cleaned the text, encoded labels, built numeric features, converted text into TF-IDF vectors, and trained three interpretable classifiers.

The best-performing model was Linear SVM, with 0.9789 accuracy and 0.9789 macro F1 on the held-out test set. The evaluation showed that ham messages were the easiest to classify, while spam and smishing were the most commonly confused classes. Overall, the project demonstrates that a well-prepared TF-IDF pipeline with classical linear models can provide strong, explainable SMS classification performance.

## References

[1] M. Munoz and M. Islam, "A Balanced Dataset for Spam and Smishing Detection using Large Language Models (LLMs)," Mendeley Data, V1, 2025, doi: 10.17632/vmg875v4xs.1. [Online]. Available: https://data.mendeley.com/datasets/vmg875v4xs/1

[2] scikit-learn developers, "sklearn.preprocessing.LabelEncoder," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

[3] scikit-learn developers, "sklearn.preprocessing.MaxAbsScaler," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html

[4] scikit-learn developers, "sklearn.feature_extraction.text.TfidfVectorizer," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

[5] G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Information Processing and Management*, vol. 24, no. 5, pp. 513-523, 1988.

[6] scikit-learn developers, "Pipelines and composite estimators," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/compose.html

[7] scikit-learn developers, "Naive Bayes," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/naive_bayes.html

[8] J. D. Rennie, L. Shih, J. Teevan, and D. R. Karger, "Tackling the poor assumptions of Naive Bayes text classifiers," in *Proc. 20th International Conference on Machine Learning (ICML)*, 2003, pp. 616-623.

[9] scikit-learn developers, "sklearn.linear_model.LogisticRegression," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

[10] scikit-learn developers, "Multiclass and multioutput algorithms," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/multiclass.html

[11] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, pp. 273-297, 1995.

[12] T. Joachims, "Text categorization with support vector machines: Learning with many relevant features," in *Proc. European Conference on Machine Learning (ECML)*, 1998, pp. 137-142.

[13] scikit-learn developers, "Support Vector Machines," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/svm.html

[14] scikit-learn developers, "Classification metrics," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/api/sklearn.metrics.html

[15] F. Pedregosa *et al*., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.
