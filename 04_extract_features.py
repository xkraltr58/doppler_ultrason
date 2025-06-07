import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import welch

def compute_features(I, Q, fs=10000):
    IQ_mag = np.sqrt(I**2 + Q**2)
    I = I - np.mean(I)
    Q = Q - np.mean(Q)

    
    freqs, psd = welch(IQ_mag, fs=fs)
    energy = np.sum(psd)
    entropy = -np.sum((psd / np.sum(psd)) * np.log2(psd / np.sum(psd) + 1e-12))
    bandwidth = np.sqrt(np.sum(((freqs - np.mean(freqs))**2) * psd) / np.sum(psd))
    peak_freq = freqs[np.argmax(psd)]

  
    iq_std = np.std(IQ_mag)
    skewness = skew(IQ_mag)
    kurt = kurtosis(IQ_mag)
    flatness = np.exp(np.mean(np.log(np.abs(IQ_mag) + 1e-8))) / (np.mean(np.abs(IQ_mag)) + 1e-8)
    zero_cross = ((IQ_mag[:-1] * IQ_mag[1:]) < 0).sum() / len(IQ_mag)

    return [
        energy,
        entropy,
        bandwidth,
        peak_freq,
        iq_std,
        skewness,
        kurt,
        flatness,
        zero_cross
    ]


data_dirs = [("Healthy Subjects", 0), ("ICU Patients", 1)]
features = []

for data_dir, label in data_dirs:
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            df = pd.read_csv(path)
            I = df["I"].values
            Q = df["Q"].values
            feats = compute_features(I, Q)
            feats.append(label)
            feats.append(filename)
            features.append(feats)


columns = [
    "MeanEnergy", "SpectralEntropy", "Bandwidth", "PeakFreq",
    "IQ_std", "Skewness", "Kurtosis", "Flatness", "ZeroCrossRate",
    "Label", "Filename"
]

df = pd.DataFrame(features, columns=columns)
df.to_csv("features.csv", index=False)
print("Yeni features.csv oluşturuldu.")
