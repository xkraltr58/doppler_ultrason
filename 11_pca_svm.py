import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


df = pd.read_csv("selected_features.csv")


X = df.drop(columns=["Label", "Filename"])
y = df["Label"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# SVM + GridSearch
param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf'], 'gamma': ['scale', 'auto']}
grid = GridSearchCV(SVC(probability=True), param_grid, cv=5)
grid.fit(X_train_res, y_train_res)

# Prediction
y_pred = grid.predict(X_test)
y_proba = grid.predict_proba(X_test)[:, 1]


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("PCA + SVM Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.show()


fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC - PCA + SVM")
plt.legend()
plt.tight_layout()
plt.show()


print("PCA + SVM Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


joblib.dump(grid.best_estimator_, "svm_pca_model.pkl")
print("Model 'svm_pca_model.pkl' olarak kaydedildi.")
