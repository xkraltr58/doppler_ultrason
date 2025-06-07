import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fft import fft
import os


normalized_file = "normalized_data.csv"  
if not os.path.exists(normalized_file):
    raise FileNotFoundError(f"{normalized_file} not found. Please run 02_normalize.py first.")

data = pd.read_csv(normalized_file)


I_norm = data["I_norm"].values
Q_norm = data["Q_norm"].values
sample_rate = 12500 


window_size = 1024  
hop_size = 512      
window_fn = get_window("hamming", window_size)

def compute_spectrogram(signal, window, hop, N_fft):
    num_frames = (len(signal) - window_size) // hop
    spectrogram = []

    for i in range(num_frames):
        start = i * hop
        end = start + window_size
        segment = signal[start:end] * window
        spectrum = np.abs(fft(segment, n=N_fft))[:N_fft // 2]
        spectrogram.append(spectrum)

    return np.array(spectrogram).T  

N_fft = window_size
spec_I = compute_spectrogram(I_norm, window_fn, hop_size, N_fft)
spec_Q = compute_spectrogram(Q_norm, window_fn, hop_size, N_fft)
frequencies = np.linspace(0, sample_rate / 2, N_fft // 2)


plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.imshow(10 * np.log10(spec_I + 1e-12), aspect='auto', origin='lower',
           extent=[0, len(I_norm)/sample_rate, 0, sample_rate/2])
plt.title("Spectrogram of Normalized I")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label='Power (dB)')

plt.subplot(2, 1, 2)
plt.imshow(10 * np.log10(spec_Q + 1e-12), aspect='auto', origin='lower',
           extent=[0, len(Q_norm)/sample_rate, 0, sample_rate/2])
plt.title("Spectrogram of Normalized Q")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label='Power (dB)')

plt.tight_layout()
plt.show()
