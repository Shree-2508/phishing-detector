"""
model_trainer.py
─────────────────────────────────────────────────────────────────────────────
PhishGuard AI — Model Training Script
Loads the Kaggle CSV (text_combined, label), augments it with
high-quality synthetic phishing samples, trains a TF-IDF +
RandomForestClassifier pipeline, evaluates it, and saves to disk.

Usage:
    python model_trainer.py --csv path/to/dataset.csv --out model/
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import csv
import json
import pickle
import random
import argparse
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# ─── SYNTHETIC PHISHING CORPUS ───────────────────────────────────────────────
# Realistic phishing templates covering Indian enterprises + global patterns.

PHISHING_TEMPLATES = [
    # SBI / Banking
    "urgent your sbi account has been suspended click here to verify account details immediately action required password otp bank",
    "dear customer your sbi netbanking access will be blocked verify your account now enter otp pin update payment",
    "sbi alert your account balance has exceeded limits verify identity now or account will be terminated",
    "hdfc bank security alert your debit card blocked suspicious transaction detected verify card details cvv expiry",
    "icici bank urgent kyc verification required submit aadhaar pan card within 24 hours avoid account suspension",
    "axis bank dear valued customer unusual login attempt detected verify your credentials immediately avoid blocking",
    "dear user your bank account will be deactivated update kyc immediately click link enter password otp",
    "your net banking password expires today update immediately to avoid losing access enter current password new password",
    # Paytm / UPI / Wallets
    "congratulations you have won rs 50000 paytm cashback claim your prize enter bank account number otp now",
    "paytm urgent your wallet has been credited with rs 10000 verify bank account claim reward limited time offer",
    "your paytm account is suspended due to suspicious activity verify mobile number otp to restore access",
    "phonepe alert your upi id will be blocked update your kyc verify aadhaar pan mobile number today",
    "google pay reward you have won cashback click here verify bank account enter upi pin claim instantly",
    # Amazon / Flipkart / Shopping
    "amazon india your account is on hold verify payment method to continue shopping credit card details required",
    "flipkart prize winner congratulations you won iphone 14 claim prize enter bank details address within 24 hours",
    "amazon your order has been cancelled refund of rs 5000 will be credited enter account details now",
    "dear amazon customer your prime membership will expire renew now to avoid losing benefits enter card details",
    # Government / Aadhaar / PAN
    "uidai aadhaar update your aadhaar card will be deactivated submit biometric otp verification link expires today",
    "income tax department notice your pan card linked accounts under scrutiny verify details avoid penalty click here",
    "government of india reward scheme you are eligible for rs 25000 subsidy submit aadhaar bank account details",
    "pm relief fund your application approved claim rs 15000 benefit enter aadhaar bank account ifsc code now",
    "epfo urgent your pf account will be suspended verify uan password mobile number aadhaar immediately",
    # Generic phishing
    "your account password will expire in 24 hours click here to update your password now urgent action required",
    "congratulations you have been selected for a free gift claim your prize by entering your personal details",
    "security alert we detected unusual login your account will be locked verify identity now",
    "you have won a lottery prize of 1000000 claim your prize by sending your bank details personal information",
    "your email account storage is full click here to upgrade and verify your credentials password username",
    "invoice attached urgent payment required immediately click here to pay outstanding balance credit card",
    "dear user your subscription expires today enter payment information credit card number cvv to renew",
    "your account has been compromised reset password immediately click link enter old password new password otp",
    "free offer limited time register now enter personal details to claim exclusive discount coupon bank transfer",
    "job offer work from home earn rs 50000 monthly click link register provide bank account salary transfer",
    "covid relief fund approved for you claim benefit enter aadhaar bank details mobile otp application form",
    "your emi payment failed update account details immediately to avoid penalty interest bank account ifsc",
    "jio offer free recharge for 1 year click here verify mobile number enter otp bank details claim reward",
    "airtel winner prize selected for free 5g upgrade verify account enter id proof address bank transfer",
    # Hindi-script mixed
    "urgent aapka bank account band ho jayega turant verify karein otp darj karein password update karein",
    "congratulations aapne rs 100000 jeeta hai claim karne ke liye bank account otp darj karein abhi",
    "aapka aadhaar kyc update required hai nahi kiya toh account suspend hoga turant link par click karein",
    "sbi alert aapke account mein suspicious transaction dekha gaya hai verify karein otp password bank",
    # Tamil-script mixed
    "urgent ungal bank account suspend seyyappadum undan aadhaar otp verify pannum link click pannungal",
    "congratulations neenga rs 50000 jithirukeenga bank account details otp enter pannungal intha offer limited time",
    # Professional phishing (spear)
    "dear employee please update your hr portal credentials immediately click link enter username password otp payroll",
    "it helpdesk your office 365 password expires today reset immediately enter current credentials new password",
    "ceo message urgent wire transfer required today send account details to finance department immediately confidential",
    "hr department your salary account update required submit bank account details form by end of day payroll",
    "your zoom meeting invitation click link enter credentials to join important board meeting urgent today",
]

LEGIT_EXTRA_TEMPLATES = [
    "team meeting scheduled for monday please find attached agenda items for the quarterly review",
    "invoice for services rendered please process payment as per terms and conditions attached",
    "your order has been shipped tracking number attached estimated delivery 3 to 5 business days",
    "welcome to our newsletter unsubscribe link at bottom we never share your email address",
    "project update attached please review and provide feedback by end of week team collaboration",
    "your monthly statement is ready login to view transaction history balance summary account",
    "reminder upcoming maintenance window saturday night systems will be unavailable briefly",
    "conference call details dial in number passcode attached calendar invite sent separately",
    "thank you for your purchase receipt attached return policy within 30 days no questions asked",
    "policy update please read updated terms of service privacy policy effective next month",
    "happy birthday team wishes you a wonderful year ahead enjoy your special day",
    "quarterly report attached performance metrics revenue growth highlights attached for review",
    "leave application approved enjoy your vacation out of office auto reply enabled dates",
    "code review completed minor suggestions see comments repository pull request merged today",
    "training session next week please register using the link below seats limited book now",
]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Pure-Python Levenshtein distance (no external library)."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[-1]


def preprocess_text(text: str) -> str:
    """Lowercase, strip URLs, numbers, special chars, normalise whitespace."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\d+', ' num ', text)
    text = re.sub(r'[^a-z\u0900-\u097f\u0b80-\u0bff\u0c00-\u0c7f ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_csv_dataset(csv_path: str):
    """Load the Kaggle CSV and return (texts, labels)."""
    texts, labels = [], []
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get('text_combined', '').strip()
            l = row.get('label', '').strip()
            if t and l in ('0', '1'):
                texts.append(preprocess_text(t))
                labels.append(int(l))
    return texts, labels


def build_augmented_dataset(csv_path: str, random_seed: int = 42):
    """
    Combine CSV data with synthetic phishing + extra legit samples.
    Returns balanced (texts, labels).
    """
    random.seed(random_seed)
    csv_texts, csv_labels = load_csv_dataset(csv_path)
    print(f"[dataset] CSV loaded: {len(csv_texts)} rows "
          f"| legit={csv_labels.count(0)}, phishing={csv_labels.count(1)}")

    # Augment phishing
    phishing_texts = [preprocess_text(t) for t in PHISHING_TEMPLATES]
    # Augment extra legit
    legit_extra = [preprocess_text(t) for t in LEGIT_EXTRA_TEMPLATES]

    all_texts = csv_texts + phishing_texts + legit_extra
    all_labels = csv_labels + [1] * len(phishing_texts) + [0] * len(legit_extra)

    # Balance: oversample minority if needed
    legit_idx = [i for i, l in enumerate(all_labels) if l == 0]
    phish_idx = [i for i, l in enumerate(all_labels) if l == 1]
    print(f"[dataset] Before balance: legit={len(legit_idx)}, phishing={len(phish_idx)}")

    max_class = max(len(legit_idx), len(phish_idx))
    if len(phish_idx) < max_class:
        extra = random.choices(phish_idx, k=max_class - len(phish_idx))
        all_texts += [all_texts[i] for i in extra]
        all_labels += [1] * len(extra)
    elif len(legit_idx) < max_class:
        extra = random.choices(legit_idx, k=max_class - len(legit_idx))
        all_texts += [all_texts[i] for i in extra]
        all_labels += [0] * len(extra)

    print(f"[dataset] After balance: total={len(all_labels)} "
          f"| legit={all_labels.count(0)}, phishing={all_labels.count(1)}")
    return all_texts, all_labels


def build_pipeline() -> Pipeline:
    """
    TF-IDF (char + word n-grams) + Soft-Voting Ensemble
    (RandomForest + GradientBoosting + LogisticRegression).
    """
    tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        max_features=15000,
        sublinear_tf=True,
        min_df=1,
        strip_accents='unicode',
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=30, min_samples_split=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=42
    )
    lr = LogisticRegression(
        C=1.0, max_iter=1000, class_weight='balanced',
        solver='lbfgs', random_state=42
    )
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft'
    )
    return Pipeline([('tfidf', tfidf), ('clf', ensemble)])


def evaluate(pipeline, X_test, y_test):
    """Print full evaluation report and return metrics dict."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "roc_auc":   round(roc_auc_score(y_test, y_prob) * 100, 2),
        "fpr":       round((confusion_matrix(y_test, y_pred)[0, 1] /
                            max(sum(1 for l in y_test if l == 0), 1)) * 100, 2),
    }
    print("\n" + "=" * 60)
    print("  PHISHGUARD MODEL EVALUATION")
    print("=" * 60)
    print(classification_report(y_test, y_pred,
                                 target_names=["Legitimate", "Phishing"]))
    print(f"  Accuracy : {metrics['accuracy']}%")
    print(f"  Precision: {metrics['precision']}%")
    print(f"  Recall   : {metrics['recall']}%")
    print(f"  F1 Score : {metrics['f1']}%")
    print(f"  ROC-AUC  : {metrics['roc_auc']}%")
    print(f"  FPR      : {metrics['fpr']}%")
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print("=" * 60)
    return metrics


def save_model(pipeline, metrics, output_dir: str):
    """Persist pipeline + metadata to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(output_dir, "phishguard_model.pkl")
    meta_path  = os.path.join(output_dir, "model_meta.json")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    with open(meta_path, "w") as f:
        json.dump({"metrics": metrics, "version": "2.0",
                   "model": "TF-IDF + VotingEnsemble(RF+GB+LR)"}, f, indent=2)
    print(f"\n[save] Model  → {model_path}")
    print(f"[save] Meta   → {meta_path}")
    return model_path


def load_model(output_dir: str):
    """Load persisted pipeline from disk."""
    model_path = os.path.join(output_dir, "phishguard_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def train(csv_path: str, output_dir: str = "model"):
    """Full training pipeline: load → augment → split → train → eval → save."""
    print(f"\n[PhishGuard Trainer] Dataset: {csv_path}")
    texts, labels = build_augmented_dataset(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42,
        stratify=labels
    )
    print(f"[split] Train={len(X_train)}  Test={len(X_test)}")

    print("\n[train] Fitting TF-IDF + Ensemble pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("[train] Done.")

    metrics = evaluate(pipeline, X_test, y_test)
    model_path = save_model(pipeline, metrics, output_dir)
    return pipeline, metrics, model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhishGuard Model Trainer")
    parser.add_argument("--csv", default="sample_dataset.csv",
                        help="Path to Kaggle phishing CSV")
    parser.add_argument("--out", default="model",
                        help="Output directory for saved model")
    args = parser.parse_args()
    train(args.csv, args.out)
