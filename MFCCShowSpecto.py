import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio file
file_path = "CopyMoveForgery.wav"
y, sr = librosa.load(file_path)

# Compute MFCCs
mfccs = librosa.feature.mfcc(y, sr)

# Display MFCC spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis="time", sr=sr)
plt.colorbar(format="%+2f")
plt.title("MFCC Spectrogram")
plt.tight_layout()
plt.show()