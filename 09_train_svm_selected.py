import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


df = pd.read_csv("selected_features.csv")
X = df.drop(columns=["Label", "Filename"])
y = df["Label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("SMOTE uygulandı.")
print("Yeni eğitim sınıf dağılımı:", np.bincount(y_train_sm))


param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(probability=True), param_grid, cv=5)
grid.fit(X_train_sm, y_train_sm)

print("En iyi parametreler:", grid.best_params_)


y_pred = grid.predict(X_test)
print("\n SMOTE + SVM (Seçilen Feature'larla) Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("SVM (Selected Features) Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show()


y_proba = grid.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='orange')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC - SVM (Selected Features)")
plt.legend()
plt.grid(True)
plt.show()


with open("svm_selected_model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)
print("Model 'svm_selected_model.pkl' olarak kaydedildi.")
