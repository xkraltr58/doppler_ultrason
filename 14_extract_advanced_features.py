import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from scipy.fftpack import fft
from scipy.signal import periodogram
from scipy.integrate import trapezoid


folders = [("Healthy Subjects", 0), ("ICU Patients", 1)]
output_file = "features_advanced.csv"
sampling_rate = 12500 

def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def spectral_flatness(signal):
    power_spectrum = np.abs(fft(signal))**2
    geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-12)))
    arithmetic_mean = np.mean(power_spectrum)
    return geometric_mean / (arithmetic_mean + 1e-12)

def zero_crossing_rate(signal):
    return np.mean(np.abs(np.diff(np.sign(signal)))) / 2

def bandpower(freqs, psd, fmin, fmax):
    mask = (freqs >= fmin) & (freqs <= fmax)
    return trapezoid(psd[mask], freqs[mask])


all_features = []


for folder, label in folders:
    for file in os.listdir(folder):
        if not file.endswith(".txt"):
            continue
        path = os.path.join(folder, file)
        df = pd.read_csv(path)
        I = normalize(df["I"].values)
        Q = normalize(df["Q"].values)
        IQ = np.sqrt(I**2 + Q**2)

        
        mean_val = np.mean(IQ)
        std_val = np.std(IQ)
        skew_val = skew(IQ)
        kurt_val = kurtosis(IQ)
        flatness = spectral_flatness(IQ)
        zcr = zero_crossing_rate(IQ)

        freqs, psd = periodogram(IQ, fs=sampling_rate)
        bp_alpha = bandpower(freqs, psd, 8, 12)
        bp_beta  = bandpower(freqs, psd, 12, 30)
        bp_gamma = bandpower(freqs, psd, 30, 45)

        all_features.append([
            mean_val, std_val, skew_val, kurt_val,
            flatness, zcr, bp_alpha, bp_beta, bp_gamma,
            label, file
        ])


columns = [
    "Mean", "Std", "Skewness", "Kurtosis",
    "SpectralFlatness", "ZeroCrossRate",
    "BandpowerAlpha", "BandpowerBeta", "BandpowerGamma",
    "Label", "Filename"
]

df_out = pd.DataFrame(all_features, columns=columns)
df_out.to_csv(output_file, index=False)
print(f"Özellik çıkarımı tamamlandı. {output_file} dosyası oluşturuldu.")
