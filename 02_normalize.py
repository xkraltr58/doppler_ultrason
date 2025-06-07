import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_dir = "Healthy Subjects"  
files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
print(f"Found {len(files)} files.")


sample_path = os.path.join(data_dir, files[0])
df = pd.read_csv(sample_path)


I = df["I"].values
Q = df["Q"].values
IQ_mag = np.sqrt(I**2 + Q**2)


def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

I_norm = normalize(I)
Q_norm = normalize(Q)
IQ_mag_norm = normalize(IQ_mag)


plt.figure(figsize=(12, 6))
plt.plot(I_norm, label='Normalized I', alpha=0.7)
plt.plot(Q_norm, label='Normalized Q', alpha=0.7)
plt.plot(IQ_mag_norm, label='Normalized |IQ|', color='black')
plt.title('Normalized I/Q Signal')
plt.xlabel('Sample Index')
plt.ylabel('Z-score')
plt.legend()
plt.tight_layout()
plt.show()


out_df = pd.DataFrame({
    "I_norm": I_norm,
    "Q_norm": Q_norm,
    "IQ_mag_norm": IQ_mag_norm
})
out_df.to_csv("normalized_data.csv", index=False)
print("Normalized data saved to normalized_data.csv.")
