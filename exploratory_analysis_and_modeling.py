# -*- coding: utf-8 -*-
#WOS-46985 Research Paper Abstract Classification

import os
import re
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ============================================================================
# STEP 0: Configuration & reproducibility
# ============================================================================

SEED = 42
np.random.seed(SEED)

DATA_PATH = r"C:\Users\Avely\Desktop\AI ASSIGNMENT\Data.xlsx"

RUN_LSTM = True      
RUN_DT_GRIDSEARCH = True 
RUN_INTERACTIVE_DEMO = True  

# ============================================================================
# STEP 1: Load data
# ============================================================================

print("=" * 60)
print("STEP 1: LOAD DATA")
print("=" * 60)

df = pd.read_excel(DATA_PATH)
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ============================================================================
# STEP 2: Data cleaning (missing values, duplicates)
# ============================================================================

print("\nMissing values:\n", df.isnull().sum())
df = df.dropna(subset=["Abstract", "Domain"])
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows. Shape now: {df.shape}")

# fill keyword for LSTM
if "keywords" in df.columns:
    df["keywords"] = df["keywords"].fillna("")
else:
    df["keywords"] = ""


# ============================================================================
# STEP 3: EDA
# ============================================================================

print("\n" + "=" * 60)
print("STEP 3: EDA")
print("=" * 60)

domain_counts = df["Domain"].value_counts()
print("\nDomain distribution:")
for domain, count in domain_counts.items():
    print(f"  {domain}: {count:,} ({count/len(df)*100:.1f}%)")

df["text_length"] = df["Abstract"].str.len()
df["word_count"] = df["Abstract"].str.split().str.len()
print(f"\nAverage abstract length: {df['text_length'].mean():.0f} characters")
print(f"Average word count: {df['word_count'].mean():.0f} words")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
domain_counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
axes[0].set_title("Domain Distribution (Target Variable)")
axes[0].tick_params(axis="x", rotation=15)
df["text_length"].hist(bins=50, ax=axes[1], color="coral", edgecolor="black")
axes[1].set_title("Abstract Length Distribution")
plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150)
plt.show()
plt.close()
print("Saved eda_overview.png")

# ============================================================================
# STEP 4: Preprocessing 
# ============================================================================

print("\n" + "=" * 60)
print("STEP 4: TEXT PREPROCESSING")
print("=" * 60)

import nltk
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """Lowercase -> strip punctuation/numbers -> tokenize -> remove stopwords -> lemmatize."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


print("Cleaning + tokenizing + lemmatizing all abstracts (this can take a few minutes)...")
df["Processed_Abstract"] = df["Abstract"].apply(preprocess_text)
print("Done. Example:")
print(" Original :", df["Abstract"].iloc[0][:150], "...")
print(" Processed:", df["Processed_Abstract"].iloc[0][:150], "...")

print("\nCleaning + tokenizing + lemmatizing keywords column...")
df["Processed_Keywords"] = df["keywords"].apply(preprocess_text)
print("Done. Example:")
print(" Original :", df["keywords"].iloc[0])
print(" Processed:", df["Processed_Keywords"].iloc[0])

# ============================================================================
# STEP 4b: Word Frequency Analysis + Word Cloud
# ============================================================================

print("\n" + "=" * 60)
print("STEP 4b: WORD FREQUENCY ANALYSIS")
print("=" * 60)

from collections import Counter

all_tokens = " ".join(df["Processed_Abstract"]).split()
word_freq = Counter(all_tokens)

print("\nTop 20 most common words (overall):")
for i, (word, count) in enumerate(word_freq.most_common(20)):
    print(f"  {i+1}. '{word}': {count:,}")

top_words = dict(word_freq.most_common(20))
plt.figure(figsize=(12, 6))
plt.bar(top_words.keys(), top_words.values(), color="steelblue", edgecolor="black")
plt.title("Top 20 Most Common Words in Abstracts")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top_words_overall.png", dpi=150)
plt.show()
plt.close()
print("Saved top_words_overall.png")

print("\nTop 10 words per Domain:")
for domain in df["Domain"].unique():
    domain_tokens = " ".join(df[df["Domain"] == domain]["Processed_Abstract"]).split()
    domain_freq = Counter(domain_tokens)
    print(f"\n  {domain}:")
    for word, count in domain_freq.most_common(10):
        print(f"    - '{word}': {count}")

try:
    from wordcloud import WordCloud
    wc = WordCloud(width=900, height=450, background_color="white",
                    max_words=100, colormap="Blues").generate(" ".join(all_tokens))
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Common Words in Abstracts")
    plt.tight_layout()
    plt.savefig("wordcloud.png", dpi=150)
    plt.show()
    plt.close()
    print("Saved wordcloud.png")
except ImportError:
    print("\n'wordcloud' package not installed. Skipping word cloud.")
    print("  To install: pip install wordcloud")

# ============================================================================
# STEP 5: Labels + shared train/test split (used by DT and LR)
# ============================================================================

print("\n" + "=" * 60)
print("STEP 5: LABEL ENCODING & TRAIN/TEST SPLIT")
print("=" * 60)

label_encoder = LabelEncoder()
df["Domain_Label"] = label_encoder.fit_transform(df["Domain"])
print(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

X_text = df["Processed_Abstract"]
X_keywords = df["Processed_Keywords"]
y = df["Domain_Label"].values

X_train_text, X_test_text, X_train_kw, X_test_kw, y_train, y_test = train_test_split(
    X_text, X_keywords, y, test_size=0.3, random_state=SEED, stratify=y
)
print(f"Train: {len(X_train_text):,}  Test: {len(X_test_text):,}")

# Shared TF-IDF vectorizer for Decision Tree & Logistic Regression
tfidf = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)
print(f"TF-IDF shape -> train {X_train_tfidf.shape}, test {X_test_tfidf.shape}")

results_summary = []  # collect metrics for the final comparison table


def evaluate_model(name, y_true, y_pred, train_time):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    results_summary.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "Train Time (s)": train_time
    })
    print(f"\n{name} -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))


def plot_confusion(name, y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(label_encoder.classes_))
    plt.xticks(ticks, label_encoder.classes_, rotation=45, ha="right")
    plt.yticks(ticks, label_encoder.classes_)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    plt.close()
    print(f"Saved {filename}")


# ============================================================================
# STEP 6: MODEL 1 - Decision Tree 
# ============================================================================

print("\n" + "=" * 60)
print("STEP 6: DECISION TREE")
print("=" * 60)

t0 = time.time()
dt = DecisionTreeClassifier(random_state=SEED)
dt.fit(X_train_tfidf, y_train)

if RUN_DT_GRIDSEARCH:
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [20, 30, None],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(DecisionTreeClassifier(random_state=SEED), param_grid,
                         cv=5, scoring="f1_weighted", n_jobs=-1)
    grid.fit(X_train_tfidf, y_train)
    dt = grid.best_estimator_
    print("Best params:", grid.best_params_)

dt_time = time.time() - t0
dt_pred = dt.predict(X_test_tfidf)
evaluate_model("Decision Tree", y_test, dt_pred, dt_time)
plot_confusion("Decision Tree", y_test, dt_pred, "confusion_decision_tree.png")

# Leak-free cross-validation using a Pipeline
dt_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)),
    ("clf", dt)
])
cv_scores_dt = cross_val_score(dt_pipeline, X_text, y, cv=5, scoring="f1_weighted", n_jobs=-1)
print(f"Decision Tree 5-fold CV F1: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std():.4f})")

# ============================================================================
# STEP 7: MODEL 2 - Logistic Regression 
# ============================================================================

print("\n" + "=" * 60)
print("STEP 7: LOGISTIC REGRESSION")
print("=" * 60)

t0 = time.time()
lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED, n_jobs=-1)
lr.fit(X_train_tfidf, y_train)
lr_time = time.time() - t0

lr_pred = lr.predict(X_test_tfidf)
evaluate_model("Logistic Regression", y_test, lr_pred, lr_time)
plot_confusion("Logistic Regression", y_test, lr_pred, "confusion_logistic_regression.png")

# ---- Feature importance: top keywords per class ----
print("\n--- Top 10 Keywords per Class (Logistic Regression coefficients) ---")
feature_names = tfidf.get_feature_names_out()
for class_idx, class_name in enumerate(label_encoder.classes_):
    coef = lr.coef_[class_idx]
    top_idx = coef.argsort()[-10:][::-1]
    words = [(feature_names[i], round(coef[i], 3)) for i in top_idx]
    print(f"\n  {class_name}:")
    for word, weight in words:
        print(f"    '{word}': {weight}")

# ---- Error analysis: misclassified samples + most confused class pairs ----
print("\n--- Error Analysis ---")
X_test_reset = X_test_text.reset_index(drop=True)
wrong_idx = np.where(lr_pred != y_test)[0]
print(f"Misclassified: {len(wrong_idx):,} / {len(y_test):,} "
      f"({len(wrong_idx)/len(y_test)*100:.2f}% error rate)")

print("\nFirst 5 misclassified samples:")
for idx in wrong_idx[:5]:
    actual = label_encoder.inverse_transform([y_test[idx]])[0]
    predicted = label_encoder.inverse_transform([lr_pred[idx]])[0]
    print(f"  Actual: {actual:20s} | Predicted: {predicted:20s} | "
          f"Text: {X_test_reset.iloc[idx][:100]}...")

confusion_pairs = Counter()
for idx in wrong_idx:
    actual = label_encoder.inverse_transform([y_test[idx]])[0]
    predicted = label_encoder.inverse_transform([lr_pred[idx]])[0]
    confusion_pairs[tuple(sorted([actual, predicted]))] += 1

print("\nMost frequently confused class pairs:")
for (a, b), count in confusion_pairs.most_common(5):
    print(f"  {a} <-> {b}: confused {count} times")

# ============================================================================
# STEP 7b: FEATURE EXTRACTION ANALYSIS 1 - N-GRAM (BIGRAM)
# ============================================================================
 
print("\n" + "=" * 60)
print("STEP 7b: N-GRAM (BIGRAM) FEATURE EXTRACTION + ACCURACY VALIDATION")
print("=" * 60)
 
tfidf_bigram = TfidfVectorizer(max_features=8000, min_df=5, max_df=0.8, ngram_range=(1, 2))
X_train_bigram = tfidf_bigram.fit_transform(X_train_text)
X_test_bigram = tfidf_bigram.transform(X_test_text)
feature_names_bigram = tfidf_bigram.get_feature_names_out()
print(f"Unigram-only vocabulary size : {X_train_tfidf.shape[1]}")
print(f"Unigram+Bigram vocabulary size: {X_train_bigram.shape[1]}")
 
lr_bigram = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED, n_jobs=-1)
lr_bigram.fit(X_train_bigram, y_train)
bigram_pred = lr_bigram.predict(X_test_bigram)
 
acc_unigram = accuracy_score(y_test, lr_pred)
f1_unigram = f1_score(y_test, lr_pred, average="weighted")
acc_bigram = accuracy_score(y_test, bigram_pred)
f1_bigram = f1_score(y_test, bigram_pred, average="weighted")
 
print(f"\nLogistic Regression (Unigram only)      -> Accuracy: {acc_unigram:.4f} | F1: {f1_unigram:.4f}")
print(f"Logistic Regression (Unigram + Bigram)  -> Accuracy: {acc_bigram:.4f} | F1: {f1_bigram:.4f}")
print(f"Accuracy change from adding bigrams: {acc_bigram - acc_unigram:+.4f}")
 
# Top bigram (2-word) phrases per Domain, from the trained LR's coefficients
print("\n--- Top Bigram (2-word) Phrases per Domain (Logistic Regression coefficients) ---")
ngram_report = []
for class_idx, class_name in enumerate(label_encoder.classes_):
    coef = lr_bigram.coef_[class_idx]
    top_idx = coef.argsort()[-60:][::-1]  # take extra, then filter to phrases only
    phrases = [(feature_names_bigram[i], round(coef[i], 3))
               for i in top_idx if " " in feature_names_bigram[i]][:10]
    print(f"\n  {class_name}:")
    for phrase, weight in phrases:
        print(f"    '{phrase}': {weight}")
        ngram_report.append({"Domain": class_name, "Bigram": phrase, "Weight": weight})
 
pd.DataFrame(ngram_report).to_csv("top_bigrams_per_domain.csv", index=False)
print("\nSaved top_bigrams_per_domain.csv")
 
ngram_comparison_df = pd.DataFrame([
    {"Feature Type": "Unigram TF-IDF (baseline)", "Vocabulary Size": X_train_tfidf.shape[1],
     "Accuracy": acc_unigram, "F1": f1_unigram},
    {"Feature Type": "Unigram + Bigram TF-IDF", "Vocabulary Size": X_train_bigram.shape[1],
     "Accuracy": acc_bigram, "F1": f1_bigram},
])
print("\n--- N-gram Comparison Summary (for Section 4.1) ---")
print(ngram_comparison_df.to_string(index=False))
ngram_comparison_df.to_csv("ngram_comparison.csv", index=False)
print("Saved ngram_comparison.csv")
 
# ============================================================================
# STEP 7c: FEATURE EXTRACTION ANALYSIS 2 - KEYWORD OVERLAP 
# ============================================================================
 
print("\n" + "=" * 60)
print("STEP 7c: KEYWORD OVERLAP (JACCARD) + CROSS-CHECK AGAINST ACTUAL CONFUSION")
print("=" * 60)
 
def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2)
 
TOP_N_KEYWORDS = 30
top_words_per_domain = {}
for class_idx, class_name in enumerate(label_encoder.classes_):
    mask = (y_train == class_idx)
    mean_tfidf_class = np.asarray(X_train_tfidf[mask].mean(axis=0)).flatten()
    top_idx = mean_tfidf_class.argsort()[-TOP_N_KEYWORDS:][::-1]
    top_words_per_domain[class_name] = set(feature_names[i] for i in top_idx)
 
print(f"\n--- Pairwise Jaccard Similarity (based on top {TOP_N_KEYWORDS} TF-IDF keywords per Domain) ---")
overlap_results = []
domain_list = list(top_words_per_domain.keys())
for i in range(len(domain_list)):
    for j in range(i + 1, len(domain_list)):
        d1, d2 = domain_list[i], domain_list[j]
        shared = top_words_per_domain[d1] & top_words_per_domain[d2]
        sim = jaccard_similarity(top_words_per_domain[d1], top_words_per_domain[d2])
        overlap_results.append({
            "Domain A": d1, "Domain B": d2,
            "Jaccard Similarity": round(sim, 3),
            "Shared Words": ", ".join(sorted(shared)) if shared else ""
        })
 
overlap_df = pd.DataFrame(overlap_results).sort_values("Jaccard Similarity", ascending=False)
print(overlap_df.to_string(index=False))
overlap_df.to_csv("domain_keyword_overlap.csv", index=False)
print("\nSaved domain_keyword_overlap.csv")
 
print("\nTop 3 most-overlapping (i.e. most confusable) Domain pairs by keyword overlap alone:")
for _, row in overlap_df.head(3).iterrows():
    print(f"  {row['Domain A']} <-> {row['Domain B']}: Jaccard = {row['Jaccard Similarity']}"
          f"  | shared words: {row['Shared Words']}")
 
print("\nCross-check: Jaccard keyword overlap vs ACTUAL misclassifications (from STEP 7):")
for (a, b), count in confusion_pairs.most_common(5):
    row = overlap_df[((overlap_df["Domain A"] == a) & (overlap_df["Domain B"] == b)) |
                      ((overlap_df["Domain A"] == b) & (overlap_df["Domain B"] == a))]
    jac = row["Jaccard Similarity"].values[0] if len(row) else "N/A"
    print(f"  {a} <-> {b}: confused {count} times in predictions | Jaccard similarity = {jac}")

# ============================================================================
# STEP 8: MODEL 3 - BiLSTM, run as an ABLATION STUDY 
# ============================================================================

if RUN_LSTM:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Embedding, LSTM, Bidirectional, Dense, Dropout, Concatenate
    )
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.utils.class_weight import compute_class_weight

    tf.random.set_seed(SEED)

    # ---- Shared tokenizer / sequence settings for both models ----
    ABS_MAX_WORDS, ABS_MAX_LENGTH = 10000, 200
    KW_MAX_WORDS, KW_MAX_LENGTH = 4000, 20

    abs_tokenizer = Tokenizer(num_words=ABS_MAX_WORDS)
    abs_tokenizer.fit_on_texts(X_train_text)  # SAME train split as DT/LR

    kw_tokenizer = Tokenizer(num_words=KW_MAX_WORDS)
    kw_tokenizer.fit_on_texts(X_train_kw)

    X_train_abs_seq = pad_sequences(abs_tokenizer.texts_to_sequences(X_train_text),
                                     maxlen=ABS_MAX_LENGTH, padding="post", truncating="post")
    X_test_abs_seq = pad_sequences(abs_tokenizer.texts_to_sequences(X_test_text),
                                    maxlen=ABS_MAX_LENGTH, padding="post", truncating="post")

    X_train_kw_seq = pad_sequences(kw_tokenizer.texts_to_sequences(X_train_kw),
                                    maxlen=KW_MAX_LENGTH, padding="post", truncating="post")
    X_test_kw_seq = pad_sequences(kw_tokenizer.texts_to_sequences(X_test_kw),
                                   maxlen=KW_MAX_LENGTH, padding="post", truncating="post")

    n_classes = len(label_encoder.classes_)
    classes_present = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes_present, y=y_train)
    class_weight = dict(zip(classes_present, weights))

    FIT_KWARGS = dict(
        epochs=10, batch_size=32, validation_split=0.2, class_weight=class_weight,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=1,
    )

    ablation_results = []  # separate table just for the single vs dual comparison

    def _record_ablation(name, y_true, y_pred, train_time):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ablation_results.append({"Model": name, "Accuracy": acc, "F1": f1,
                                  "Train Time (s)": train_time})

    # ------------------------------------------------------------------
    # Model 3a: SINGLE-INPUT BiLSTM (Abstract only) -- the baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8a: SINGLE-INPUT BiLSTM (Abstract only) -- baseline")
    print("=" * 60)

    abs_input_a = Input(shape=(ABS_MAX_LENGTH,), name="abstract_input")
    abs_emb_a = Embedding(ABS_MAX_WORDS, 128, mask_zero=True, name="abstract_embedding")(abs_input_a)
    abs_branch_a = Bidirectional(LSTM(128), name="abstract_bilstm")(abs_emb_a)
    x_a = Dense(64, activation="relu")(abs_branch_a)
    x_a = Dropout(0.5)(x_a)
    output_a = Dense(n_classes, activation="softmax")(x_a)

    model_single = Model(inputs=abs_input_a, outputs=output_a, name="single_input_bilstm")
    model_single.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model_single.summary()

    t0 = time.time()
    history_single = model_single.fit(X_train_abs_seq, y_train, **FIT_KWARGS)
    single_time = time.time() - t0

    single_pred = np.argmax(model_single.predict(X_test_abs_seq), axis=1)
    evaluate_model("BiLSTM (Abstract only)", y_test, single_pred, single_time)
    plot_confusion("BiLSTM (Abstract only)", y_test, single_pred, "confusion_lstm_single.png")
    _record_ablation("BiLSTM - Abstract only", y_test, single_pred, single_time)

    # ------------------------------------------------------------------
    # Model 3b: DUAL-INPUT BiLSTM (Abstract + Keywords)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8b: DUAL-INPUT BiLSTM (Abstract + Keywords)")
    print("=" * 60)

    abs_input_b = Input(shape=(ABS_MAX_LENGTH,), name="abstract_input")
    abs_emb_b = Embedding(ABS_MAX_WORDS, 128, mask_zero=True, name="abstract_embedding")(abs_input_b)
    abs_branch_b = Bidirectional(LSTM(128), name="abstract_bilstm")(abs_emb_b)

    kw_input_b = Input(shape=(KW_MAX_LENGTH,), name="keywords_input")
    kw_emb_b = Embedding(KW_MAX_WORDS, 32, mask_zero=True, name="keywords_embedding")(kw_input_b)
    kw_branch_b = Bidirectional(LSTM(32), name="keywords_bilstm")(kw_emb_b)

    merged_b = Concatenate(name="concat_abstract_keywords")([abs_branch_b, kw_branch_b])
    x_b = Dense(64, activation="relu")(merged_b)
    x_b = Dropout(0.5)(x_b)
    output_b = Dense(n_classes, activation="softmax")(x_b)

    model_dual = Model(inputs=[abs_input_b, kw_input_b], outputs=output_b, name="dual_input_bilstm")
    model_dual.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model_dual.summary()

    t0 = time.time()
    history_dual = model_dual.fit([X_train_abs_seq, X_train_kw_seq], y_train, **FIT_KWARGS)
    dual_time = time.time() - t0

    dual_pred = np.argmax(model_dual.predict([X_test_abs_seq, X_test_kw_seq]), axis=1)
    evaluate_model("BiLSTM (Abstract + Keywords)", y_test, dual_pred, dual_time)
    plot_confusion("BiLSTM (Abstract + Keywords)", y_test, dual_pred, "confusion_lstm_dual.png")
    _record_ablation("BiLSTM - Abstract + Keywords", y_test, dual_pred, dual_time)

    # ------------------------------------------------------------------
    # Ablation comparison table: does the keywords branch actually help?
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8c: ABLATION -- Single-input vs Dual-input BiLSTM")
    print("=" * 60)

    ablation_df = pd.DataFrame(ablation_results)
    f1_delta = ablation_df.loc[ablation_df["Model"].str.contains("Keywords"), "F1"].values[0] - \
               ablation_df.loc[ablation_df["Model"].str.contains("Abstract only"), "F1"].values[0]
    print(ablation_df.to_string(index=False))
    print(f"\nF1 change from adding the keywords branch: {f1_delta:+.4f}")
    if f1_delta > 0:
        print("-> The keywords branch improved performance. Keep the dual-input model.")
    else:
        print("-> The keywords branch did NOT improve performance on this split. Worth "
              "reporting honestly in Discussion -- e.g. the abstract may already contain "
              "enough signal, or the keywords branch needs more tuning/data.")
    ablation_df.to_csv("lstm_ablation_single_vs_dual.csv", index=False)
    print("Saved lstm_ablation_single_vs_dual.csv")

    # Training curves for both models side by side
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history_single.history["accuracy"], label="Train")
    axes[0, 0].plot(history_single.history["val_accuracy"], label="Validation")
    axes[0, 0].set_title("Single-input: Accuracy"); axes[0, 0].legend()
    axes[0, 1].plot(history_single.history["loss"], label="Train")
    axes[0, 1].plot(history_single.history["val_loss"], label="Validation")
    axes[0, 1].set_title("Single-input: Loss"); axes[0, 1].legend()
    axes[1, 0].plot(history_dual.history["accuracy"], label="Train")
    axes[1, 0].plot(history_dual.history["val_accuracy"], label="Validation")
    axes[1, 0].set_title("Dual-input: Accuracy"); axes[1, 0].legend()
    axes[1, 1].plot(history_dual.history["loss"], label="Train")
    axes[1, 1].plot(history_dual.history["val_loss"], label="Validation")
    axes[1, 1].set_title("Dual-input: Loss"); axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig("lstm_training_curves.png", dpi=150)
    plt.show()
    plt.close()

    # Keep variable names used later in the script pointing at the better-performing model
    model = model_dual if f1_delta > 0 else model_single
else:
    print("\nSkipping LSTM (RUN_LSTM=False)")

# ============================================================================
# STEP 9: FINAL MODEL COMPARISON 
# ============================================================================

print("\n" + "=" * 60)
print("STEP 9: FINAL MODEL COMPARISON")
print("=" * 60)

summary_df = pd.DataFrame(results_summary).sort_values("F1", ascending=False)
print(summary_df.to_string(index=False))
summary_df.to_csv("model_comparison_results.csv", index=False)
print("\nSaved model_comparison_results.csv")

# ============================================================================
# STEP 10: Demo (manual use only, does NOT run during grading)
# ============================================================================

MIN_WORDS_FOR_RELIABLE_PREDICTION = 20

# Was the winning LSTM model the dual-input (Abstract + Keywords) one?
# (Only meaningful when RUN_LSTM is True; STEP 8c sets `model` and `f1_delta`.)
USE_LSTM_DEMO = RUN_LSTM
USE_KEYWORDS_DEMO = RUN_LSTM and (f1_delta > 0 if RUN_LSTM else False)


def predict_category(text, keywords="", top_n=3):
    """Predict the Domain of a new abstract.

    Uses the best LSTM model from the STEP 8c ablation when available
    (dual-input BiLSTM if it won, otherwise single-input BiLSTM), and falls
    back to the Logistic Regression + TF-IDF model when RUN_LSTM=False.

    `keywords` is optional and is only used when the dual-input BiLSTM won
    the ablation -- otherwise it is silently ignored.

    If the input is too short OR doesn't match the model's vocabulary at all,
    this function prints ONLY a warning and returns None -- it does NOT print
    a (meaningless) predicted category in that case.
    """
    word_count = len(text.split())
    if word_count < MIN_WORDS_FOR_RELIABLE_PREDICTION:
        print(f"\n⚠ Warning: input has only {word_count} word(s). This model was trained on "
              f"full abstracts (~200 words). Short phrases, names, or single keywords often "
              f"produce an unreliable / meaningless prediction. Please paste a full abstract "
              f"for a trustworthy result.")
        return None, None

    if USE_LSTM_DEMO:
        # ---- LSTM path (single-input or dual-input, whichever won STEP 8c) ----
        cleaned = preprocess_text(text)
        abs_seq = pad_sequences(abs_tokenizer.texts_to_sequences([cleaned]),
                                 maxlen=ABS_MAX_LENGTH, padding="post", truncating="post")

        if abs_tokenizer.texts_to_sequences([cleaned])[0] == []:
            print("⚠ Warning: none of the words in your input matched the model's vocabulary. "
                  "No prediction is shown because it would be meaningless -- please enter a "
                  "real academic abstract.")
            return None, None

        if USE_KEYWORDS_DEMO:
            cleaned_kw = preprocess_text(keywords)
            kw_seq = pad_sequences(kw_tokenizer.texts_to_sequences([cleaned_kw]),
                                    maxlen=KW_MAX_LENGTH, padding="post", truncating="post")
            proba = model.predict([abs_seq, kw_seq], verbose=0)[0]
        else:
            proba = model.predict(abs_seq, verbose=0)[0]
    else:
        # ---- Logistic Regression fallback (RUN_LSTM=False) ----
        cleaned = preprocess_text(text)
        vec = tfidf.transform([cleaned])

        if vec.nnz == 0:
            print("⚠ Warning: none of the words in your input matched the model's vocabulary "
                  "(TF-IDF vector is all zeros). No prediction is shown because it would be "
                  "meaningless -- please enter a real academic abstract.")
            return None, None

        proba = lr.predict_proba(vec)[0]

    pred = proba.argmax()
    label = label_encoder.inverse_transform([pred])[0]
    confidence = proba[pred] * 100

    print("\n" + "-" * 60)
    print(f"Input (first 100 chars): {text[:100]}...")
    if USE_KEYWORDS_DEMO and keywords.strip():
        print(f"Keywords used: {keywords[:100]}")
    print(f"\n>>> Predicted Domain : {label}")
    print(f">>> Confidence       : {confidence:.2f}%")

    top_idx = proba.argsort()[-top_n:][::-1]
    print(f"\nTop {top_n} Most Likely Domains:")
    for rank, idx in enumerate(top_idx, start=1):
        cat = label_encoder.inverse_transform([idx])[0]
        print(f"  {rank}. {cat:20s} - {proba[idx]*100:.2f}%")

    return label, confidence


if __name__ == "__main__" and RUN_INTERACTIVE_DEMO:
    print("\n" + "=" * 60)
    print("INTERACTIVE PREDICTION DEMO")
    if USE_LSTM_DEMO:
        which = "Dual-input BiLSTM (Abstract + Keywords)" if USE_KEYWORDS_DEMO else "Single-input BiLSTM (Abstract only)"
        print(f"Using model: {which}")
    else:
        print("Using model: Logistic Regression (Abstract only)")
    print("Enter an English abstract to predict its academic Domain.")
    if USE_KEYWORDS_DEMO:
        print("You will also be asked for optional keywords (press Enter to skip).")
    print("Type 'exit' to quit.")
    print("=" * 60)

    # Quick sanity check with a real test-set example first
    print("\n--- Example: real abstract from the test set ---")
    predict_category(X_test_reset.iloc[0])

    while True:
        user_input = input("\nEnter an abstract (or 'exit' to quit): \n> ")
        if user_input.strip().lower() == "exit":
            print("Demo ended.")
            break
        if len(user_input.strip()) < 5:
            print("Input too short, please enter a fuller abstract.")
            continue

        user_keywords = ""
        if USE_KEYWORDS_DEMO:
            user_keywords = input("Enter keywords, semicolon-separated (optional, press Enter to skip): \n> ")

        predict_category(user_input, keywords=user_keywords)

print("\nPIPELINE COMPLETE")


