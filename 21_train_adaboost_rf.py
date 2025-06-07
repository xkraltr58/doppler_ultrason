import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("selected_features_advanced_cleaned.csv")


X = df.drop(["Label", "Filename"], axis=1)
y = df["Label"]


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"SMOTE uygulandı. Yeni dağılım: {dict(pd.Series(y_resampled).value_counts())}")


X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)


base_rf = RandomForestClassifier(random_state=42)


ada_rf = AdaBoostClassifier(estimator=base_rf, random_state=42)


param_grid = {
    "n_estimators": [50, 100],
    "learning_rate": [0.01, 0.1],
    "estimator__n_estimators": [100, 200],
    "estimator__max_depth": [3, None],
    "estimator__min_samples_split": [2, 5],
}

grid = GridSearchCV(ada_rf, param_grid, cv=3, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"En iyi parametreler: {grid.best_params_}")


y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]


print("\n📊 AdaBoost+RF Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


with open("adaboost_rf_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("\n Model 'adaboost_rf_model.pkl' olarak kaydedildi.")


fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - AdaBoost + RF")
plt.legend(loc="lower right")
plt.savefig("roc_adaboost_rf.png")
plt.close()


cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - AdaBoost + RF")
plt.savefig("cm_adaboost_rf.png")
plt.close()
