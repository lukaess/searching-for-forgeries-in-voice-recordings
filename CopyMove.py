import librosa
import hashlib
import numpy as np
import matplotlib.pyplot as plt

# Load the audio signal
signal, sr = librosa.load('blues.00000.wav', sr=None)

# Divide the signal into frames
frame_size = int(sr * 0.025) # 25 ms
hop_size = int(sr * 0.010)   # 10 ms
frames = librosa.util.frame(signal, frame_length=frame_size, hop_length=hop_size)

# Compute the spectrogram for each frame
spectrograms = []
for frame in frames.T:
    spectrogram = np.abs(librosa.stft(frame))
    spectrograms.append(spectrogram)

# Compute the fingerprint for each frame
fingerprints = []
for spectrogram in spectrograms:
    quantized = np.round(spectrogram / 10) # Quantize the coefficients
    hash_value = hashlib.sha256(quantized.tobytes()).hexdigest() # Hash the sequence
    fingerprints.append(hash_value)

# Compare the fingerprints to detect copy-move forgeries
duplicated_frames = []
for i in range(len(fingerprints)):
    for j in range(i+1, len(fingerprints)):
        if fingerprints[i] == fingerprints[j]:
            duplicated_frames.append((i, j))
            print("The audio has not been edited.")
            break

# Plot the spectrogram and highlight the duplicated segments
plt.figure(figsize=(10, 4))
plt.specgram(signal, Fs=sr, NFFT=2048, noverlap=512, cmap='jet')
for i, j in duplicated_frames:
    plt.axvspan(i*hop_size/sr, (i+frame_size)*hop_size/sr, facecolor='r', alpha=0.5)
    plt.axvspan(j*hop_size/sr, (j+frame_size)*hop_size/sr, facecolor='g', alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram of the audio signal')
plt.show()