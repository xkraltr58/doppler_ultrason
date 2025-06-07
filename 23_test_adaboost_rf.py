import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split


df = pd.read_csv("selected_features_advanced_cleaned.csv")


X = df.drop(columns=["Label", "Filename"])
y = df["Label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


with open("adaboost_rf_model.pkl", "rb") as f:
    model = pickle.load(f)


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


print(" AdaBoost + RF Test Set Performansı:\n")
print(classification_report(y_test, y_pred))


fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AdaBoost + RF - ROC Curve (Test Set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("test_roc_adaboost_rf.png")
plt.close()


cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("AdaBoost + RF - Confusion Matrix (Test Set)")
plt.colorbar()
plt.xticks([0, 1], ['0', '1'])
plt.yticks([0, 1], ['0', '1'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("test_confusion_adaboost_rf.png")
plt.close()

print("Test sonuçları ve grafikler başarıyla kaydedildi.")
