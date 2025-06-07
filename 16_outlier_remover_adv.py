import pandas as pd
import numpy as np
from scipy.stats import zscore


df = pd.read_csv("selected_features_advanced.csv")


features = df.drop(columns=["Label", "Filename"])
z_scores = np.abs(zscore(features))


mask = (z_scores < 3).all(axis=1)
df_cleaned = df[mask]


output_path = "selected_features_advanced_cleaned.csv"
df_cleaned.to_csv(output_path, index=False)

print(f" {len(df) - len(df_cleaned)} adet uç değer veri çıkarıldı.")
print(f" Kalan veri sayısı: {len(df_cleaned)}")
print(f" Temizlenmiş veri kaydedildi: {output_path}")

