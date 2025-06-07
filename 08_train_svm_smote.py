import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib


df = pd.read_csv("features.csv")
X = df[["MeanEnergy", "SpectralEntropy", "Bandwidth", "PeakFreq"]]
y = df["Label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("SMOTE uygulandı.")
print("Yeni eğitim sınıf dağılımı:", np.bincount(y_train_smote))


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)


param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}
grid_search = GridSearchCV(SVC(probability=True, class_weight='balanced'), param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train_smote)
print("🔍 En iyi parametreler:", grid_search.best_params_)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print("\n📊 SMOTE + SVM Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="Blues")
plt.title("SMOTE + SVM Confusion Matrix (Test Verisi)")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.show()


y_scores = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC - SMOTE + SVM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


joblib.dump(best_model, "svm_smote_model.pkl")
print("Model 'svm_smote_model.pkl' olarak kaydedildi.")
