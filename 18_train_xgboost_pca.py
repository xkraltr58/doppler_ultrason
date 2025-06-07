import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv("pca_features.csv")


X = df.drop(columns=["Label", "Filename"])
y = df["Label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [2, 3, 4]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid = GridSearchCV(xgb, param_grid, cv=5, verbose=0)
grid.fit(X_train, y_train)

print(f"\nEn iyi parametreler: {grid.best_params_}")


y_pred = grid.predict(X_test)
print("\n PCA + XGBoost Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - PCA + XGBoost")
plt.colorbar()
plt.xticks([0, 1], ['0', '1'])
plt.yticks([0, 1], ['0', '1'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_xgb_pca.png")
plt.close()


y_probs = grid.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color='orange')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - PCA + XGBoost")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_xgb_pca.png")
plt.close()


with open("xgb_pca_model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)

print("\nModel 'xgb_pca_model.pkl' olarak kaydedildi.")
