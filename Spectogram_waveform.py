import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
file_path = '018aa829-343e1ffe.wav'
audio_data, sample_rate = librosa.load(file_path)

# Calculate the Short-Time Fourier Transform (STFT) to create the spectrogram
spectrogram = librosa.stft(audio_data)

# Display the waveform
plt.figure(figsize=(12, 6))
librosa.display.waveshow(audio_data, sr=sample_rate)
plt.title('Waveform')
plt.show()
# Display the spectrogram
plt.figure(figsize=(12, 6))
spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()