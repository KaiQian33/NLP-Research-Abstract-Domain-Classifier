# -*- coding: utf-8 -*-
"""
Training pipeline. Run this ONCE (or whenever the model changes).
It saves every artifact the Streamlit app needs into ./models/,
so app.py never has to retrain anything.

Usage:
    python train.py --data Data.xlsx
"""

import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nlp_utils import preprocess_text

SEED = 42
np.random.seed(SEED)
MODEL_DIR = "models"


def main(data_path: str, run_lstm: bool = False):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---- 1. Load & clean ----
    df = pd.read_excel(data_path)
    df = df.dropna(subset=["Abstract", "Domain"]).drop_duplicates()
    if "keywords" in df.columns:
        df["keywords"] = df["keywords"].fillna("")
    else:
        df["keywords"] = ""

    # ---- 2. Preprocess ----
    print("Preprocessing abstracts (this takes a few minutes)...")
    df["Processed_Abstract"] = df["Abstract"].apply(preprocess_text)
    df["Processed_Keywords"] = df["keywords"].apply(preprocess_text)

    # ---- 3. Labels & split ----
    label_encoder = LabelEncoder()
    df["Domain_Label"] = label_encoder.fit_transform(df["Domain"])

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["Processed_Abstract"], df["Domain_Label"].values,
        test_size=0.3, random_state=SEED, stratify=df["Domain_Label"].values
    )

    tfidf = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    results = []

    def evaluate(name, model, X_test, y_test):
        pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, pred, average="weighted", zero_division=0),
        })

    # ---- 4. Decision Tree ----
    print("Training Decision Tree...")
    t0 = time.time()
    dt = DecisionTreeClassifier(random_state=SEED, max_depth=30)
    dt.fit(X_train_tfidf, y_train)
    print(f"  done in {time.time()-t0:.1f}s")
    evaluate("Decision Tree", dt, X_test_tfidf, y_test)

    # ---- 5. Logistic Regression ----
    print("Training Logistic Regression...")
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED, n_jobs=-1)
    lr.fit(X_train_tfidf, y_train)
    print(f"  done in {time.time()-t0:.1f}s")
    evaluate("Logistic Regression", lr, X_test_tfidf, y_test)

    # ---- 6. Save everything the app needs ----
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.joblib"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    joblib.dump(dt, os.path.join(MODEL_DIR, "dt_model.joblib"))
    joblib.dump(lr, os.path.join(MODEL_DIR, "lr_model.joblib"))

    summary_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    summary_df.to_csv(os.path.join(MODEL_DIR, "model_comparison_results.csv"), index=False)
    print("\n", summary_df.to_string(index=False))

    # ---- 7. (Optional) BiLSTM — heavier, off by default ----
    if run_lstm:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout

        ABS_MAX_WORDS, ABS_MAX_LENGTH = 10000, 200
        abs_tokenizer = Tokenizer(num_words=ABS_MAX_WORDS)
        abs_tokenizer.fit_on_texts(X_train_text)

        X_train_seq = pad_sequences(abs_tokenizer.texts_to_sequences(X_train_text), maxlen=ABS_MAX_LENGTH)
        X_test_seq = pad_sequences(abs_tokenizer.texts_to_sequences(X_test_text), maxlen=ABS_MAX_LENGTH)

        n_classes = len(label_encoder.classes_)
        inp = Input(shape=(ABS_MAX_LENGTH,))
        x = Embedding(ABS_MAX_WORDS, 128, mask_zero=True)(inp)
        x = Bidirectional(LSTM(128))(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.5)(x)
        out = Dense(n_classes, activation="softmax")(x)
        lstm_model = Model(inp, out)
        lstm_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        lstm_model.fit(X_train_seq, y_train, epochs=10, batch_size=32, validation_split=0.2)

        pred = np.argmax(lstm_model.predict(X_test_seq), axis=1)
        evaluate_dummy = {
            "Model": "BiLSTM (Abstract only)",
            "Accuracy": accuracy_score(y_test, pred),
            "F1": f1_score(y_test, pred, average="weighted"),
        }
        print(evaluate_dummy)

        lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.keras"))
        joblib.dump(abs_tokenizer, os.path.join(MODEL_DIR, "abs_tokenizer.joblib"))
        joblib.dump({"max_length": ABS_MAX_LENGTH}, os.path.join(MODEL_DIR, "lstm_config.joblib"))

    print(f"\nAll artifacts saved to ./{MODEL_DIR}/ — app.py can now run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Data.xlsx", help="Path to Data.xlsx")
    parser.add_argument("--run_lstm", action="store_true", help="Also train & save the BiLSTM (slow)")
    args = parser.parse_args()
    main(args.data, run_lstm=args.run_lstm)
