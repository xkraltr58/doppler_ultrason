import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize


df = pd.read_csv("selected_features_advanced_cleaned.csv")
X = df.drop(columns=["Label", "Filename"])
y = df["Label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}
grid = GridSearchCV(AdaBoostClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"En iyi parametreler: {grid.best_params_}")


y_pred = best_model.predict(X_test)
print("\n AdaBoost Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


with open("adaboost_advanced_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("\nModel 'adaboost_advanced_model.pkl' olarak kaydedildi.")


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix - AdaBoost (Advanced Features)")
plt.savefig("adaboost_confusion_matrix.png")
plt.close()


y_test_bin = label_binarize(y_test, classes=[0, 1]).ravel()
y_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - AdaBoost (Advanced Features)")
plt.legend()
plt.grid(True)
plt.savefig("adaboost_roc_curve.png")
plt.close()

print("Grafikler kaydedildi: 'adaboost_confusion_matrix.png' ve 'adaboost_roc_curve.png'")
