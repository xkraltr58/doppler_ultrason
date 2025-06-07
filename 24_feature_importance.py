import pandas as pd
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv("selected_features_advanced_cleaned.csv")
feature_names = df.drop(columns=["Label", "Filename"]).columns


with open("rf_advanced_model.pkl", "rb") as f:
    model = pickle.load(f)


importances = model.feature_importances_
sorted_indices = importances.argsort()[::-1]


plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_indices], align='center')
plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=45)
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
print(" Özellik önem grafiği kaydedildi: rf_feature_importance.png")
