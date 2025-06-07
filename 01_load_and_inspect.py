import os
import numpy as np
import matplotlib.pyplot as plt


data_dir = "Healthy Subjects"  
full_path = os.path.join(os.getcwd(), data_dir)
files = [f for f in os.listdir(full_path) if f.endswith('.txt')]

if not files:
    print(f"No .txt files found in: {full_path}")
    exit()


sample_path = os.path.join(full_path, files[0])
print(f"Loading: {sample_path}")


try:
    raw_data = np.loadtxt(sample_path, delimiter=",", skiprows=1, usecols=(1, 2))
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

I = raw_data[:, 0]
Q = raw_data[:, 1]
IQ_mag = np.sqrt(I**2 + Q**2)


plt.figure(figsize=(12, 6))
plt.plot(I, label='I', alpha=0.7)
plt.plot(Q, label='Q', alpha=0.7)
plt.plot(IQ_mag, label='|IQ|', color='black')
plt.title('Raw I/Q Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()


print("\nSignal Stats:")
print(f"I: min={I.min()}, max={I.max()}, mean={I.mean():.2f}, std={I.std():.2f}")
print(f"Q: min={Q.min()}, max={Q.max()}, mean={Q.mean():.2f}, std={Q.std():.2f}")
print(f"|IQ|: mean={IQ_mag.mean():.2f}, std={IQ_mag.std():.2f}")
print("\nNaN Check:")
print(f"I: {np.isnan(I).any()}, Q: {np.isnan(Q).any()}")
