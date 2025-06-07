import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df = pd.read_csv("selected_features_advanced_cleaned.csv")


X = df.drop(columns=["Label", "Filename"])
y = df["Label"]
filenames = df["Filename"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA()
X_pca = pca.fit_transform(X_scaled)


plt.figure()
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title("PCA Explained Variance Ratio")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_variance_plot.png")
plt.close()


n_components = next(i for i, total in enumerate(
    pca.explained_variance_ratio_.cumsum()) if total >= 0.95) + 1
print(f"%95 varyans için gerekli bileşen sayısı: {n_components}")


pca = PCA(n_components=n_components)
X_pca_final = pca.fit_transform(X_scaled)


pca_columns = [f"PCA_{i+1}" for i in range(n_components)]
pca_df = pd.DataFrame(X_pca_final, columns=pca_columns)
pca_df["Label"] = y.values
pca_df["Filename"] = filenames.values


pca_df.to_csv("pca_features.csv", index=False)
print(" PCA sonucu kaydedildi: pca_features.csv")
