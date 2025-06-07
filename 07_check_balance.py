import pandas as pd

df = pd.read_csv("features.csv")
class_counts = df["Label"].value_counts()
print("🔍 Sınıf Dağılımı:")
print(class_counts)

healthy_count = class_counts.get(0, 0)
icu_count = class_counts.get(1, 0)
print(f"\nSağlıklı: {healthy_count} örnek")
print(f"ICU: {icu_count} örnek")

ratio = round(healthy_count / icu_count, 2) if icu_count > 0 else 0
print(f"Oran (Healthy / ICU): {ratio}")
