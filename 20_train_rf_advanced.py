import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("selected_features_advanced_cleaned.csv")


X = df.drop(columns=["Label", "Filename"])
y = df["Label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("\U0001F50D En iyi parametreler:", grid.best_params_)


y_pred = grid.predict(X_test)
print("\n\U0001F4CA Random Forest Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


y_proba = grid.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest (Advanced Features)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_rf_advanced.png")
plt.close()


cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest (Advanced Features)")
plt.colorbar()
plt.xticks([0, 1], ['0', '1'])
plt.yticks([0, 1], ['0', '1'])
plt.xlabel("Predicted label")
plt.ylabel("True label")


for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig("confusion_rf_advanced.png")
plt.close()


with open("rf_advanced_model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)

print("\n\u2705 Model 'rf_advanced_model.pkl' olarak kaydedildi.")
