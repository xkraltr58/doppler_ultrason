import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np


df = pd.read_csv("features.csv")
X = df.drop(columns=["Label", "Filename"])
y = df["Label"]


k = 5
selector = SelectKBest(score_func=f_classif, k=k)
X_new = selector.fit_transform(X, y)


selected_mask = selector.get_support()
selected_features = X.columns[selected_mask]
print("Seçilen özellikler:", list(selected_features))


df_selected = pd.DataFrame(X_new, columns=selected_features)
df_selected["Label"] = y
df_selected["Filename"] = df["Filename"]
df_selected.to_csv("selected_features.csv", index=False)
print("Yeni dosya: selected_features.csv")
