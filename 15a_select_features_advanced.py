import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

warnings.filterwarnings("ignore")


df = pd.read_csv("features_advanced.csv")


X = df.drop(columns=["Label", "Filename"])
y = df["Label"]


selector = SelectKBest(score_func=f_classif, k=6)
X_new = selector.fit_transform(X, y)


selected_columns = X.columns[selector.get_support()]
print("Seçilen en iyi özellikler:", list(selected_columns))


X_selected = df[selected_columns]
X_selected["Label"] = y
X_selected["Filename"] = df["Filename"]


X_selected.to_csv("selected_features_advanced.csv", index=False)
print("💾 Kaydedildi: selected_features_advanced.csv")
