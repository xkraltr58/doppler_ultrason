import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


model_paths = {
    "SVM": "svm_advanced_model.pkl",
    "Random Forest": "rf_advanced_model.pkl",
    "AdaBoost": "adaboost_advanced_model.pkl",
    "XGBoost (PCA)": "xgb_pca_model.pkl",
    "AdaBoost + RF": "adaboost_rf_model.pkl",
}


adv_data = pd.read_csv("selected_features_advanced_cleaned.csv")
X_adv = adv_data.drop(columns=["Label", "Filename"])
y_adv = adv_data["Label"]


pca_data = pd.read_csv("pca_features.csv")
X_pca = pca_data.drop(columns=["Label", "Filename"])
y_pca = pca_data["Label"]


results = []


for name, path in model_paths.items():
    with open(path, "rb") as f:
        model = pickle.load(f)

    if "PCA" in name:
        X = X_pca
        y = y_pca
    else:
        X = X_adv
        y = y_adv

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_prob = y_pred  

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_prob),
    })


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1 Score", ascending=False)


results_df.to_csv("model_comparison_table.csv", index=False)
print("\n📊 Karşılaştırma tablosu kaydedildi: model_comparison_table.csv")
print(results_df)
