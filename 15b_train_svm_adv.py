import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv("selected_features_advanced.csv")
X = df.drop(columns=["Label", "Filename"])
y = df["Label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"SMOTE uygulandı.\nYeni eğitim sınıf dağılımı: {pd.Series(y_resampled).value_counts().to_dict()}")


param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf'],
    'class_weight': [None, 'balanced']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, refit=True, verbose=0)
grid.fit(X_resampled, y_resampled)

print("En iyi parametreler:", grid.best_params_)


y_pred = grid.predict(X_test)
print("\n📊 Gelişmiş Feature'larla SVM Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


y_prob = grid.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Advanced Features")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_advanced_features.png")
plt.close()


cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Advanced Features")
plt.colorbar()
plt.xticks([0, 1], ['0', '1'])
plt.yticks([0, 1], ['0', '1'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_advanced_features.png")
plt.close()


with open("svm_advanced_model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)

print("Model 'svm_advanced_model.pkl' olarak kaydedildi.")
