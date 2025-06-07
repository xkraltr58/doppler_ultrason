import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore


file_path = r"C:\Users\idild\OneDrive\Masaüstü\dopplerultrason\selected_features.csv"


df = pd.read_csv(file_path)


features = ['MeanEnergy', 'IQ_std', 'Skewness', 'Kurtosis', 'Flatness']


z_scores = np.abs(zscore(df[features]))


threshold = 3
mask = (z_scores < threshold).all(axis=1)

df_cleaned = df[mask]

print(f"{len(df) - len(df_cleaned)} adet uç değer veri çıkarıldı.")
print(f"Kalan veri sayısı: {len(df_cleaned)}")


cleaned_path = r"C:\Users\idild\OneDrive\Masaüstü\dopplerultrason\selected_features_cleaned.csv"
df_cleaned.to_csv(cleaned_path, index=False)
print(f"Temizlenmiş veri kaydedildi: {cleaned_path}")
