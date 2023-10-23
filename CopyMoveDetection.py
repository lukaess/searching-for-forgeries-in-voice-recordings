import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft

# Učitavanje audio zapisa
original_audio_zapis = 'SpeechSynthOrginal.wav'
krivotvoren_audio_zapis = 'SpeechSynthesis.wav'

# Uz pomoć librose se učitava audio
original_audio, original_stopa = librosa.load(original_audio_zapis)
krivotvoren_audio, neovlastena_stopa = librosa.load(krivotvoren_audio_zapis)

# Računanje the Short-time Fourier Transformacije (STFT)
D_original = librosa.stft(original_audio)
D_krivotvoren = librosa.stft(krivotvoren_audio)

# Konvertiranje vrijednosti signala, amplitude u decibele, relativna vrijednost u maksimum
D_original_db = librosa.amplitude_to_db(np.abs(D_original), ref=np.max)
D_krivotvoren_db = librosa.amplitude_to_db(np.abs(D_krivotvoren), ref=np.max)

# Račuanje MFCC značajki
mfcc_original = librosa.feature.mfcc(y=original_audio, sr=original_stopa, n_mfcc=13)
mfcc_krivotvoren = librosa.feature.mfcc(y=krivotvoren_audio, sr=neovlastena_stopa, n_mfcc=13)

# Računanje Discrete Fourier transformacije (DFT)
dft_original = np.abs(fft(original_audio)[:len(original_audio)//2])
dft_krivotvoren = np.abs(fft(krivotvoren_audio)[:len(krivotvoren_audio) // 2])

# Računaj chroma značajke
chroma_original = librosa.feature.chroma_stft(original_audio, sr=original_stopa)
chroma_krivotvoren = librosa.feature.chroma_stft(krivotvoren_audio, sr=neovlastena_stopa)

# Generiranje  chromagrams
C_original = librosa.feature.chroma_cqt(original_audio, sr=original_stopa)
C_krivotvoren = librosa.feature.chroma_cqt(krivotvoren_audio, sr=neovlastena_stopa)


# postavi
plt.figure(figsize=(12, 16))

# Original audio spectrogram
plt.subplot(5, 2, 1)
librosa.display.specshow(D_original_db, sr=original_stopa, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spektrogram Originalnog Audio Zapisa')

# krivotvorenog audio spectrogram
plt.subplot(5, 2, 2)
librosa.display.specshow(D_krivotvoren_db, sr=neovlastena_stopa, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spektrogram Krivotvorenog Audio Zapisa')

# Original audio MFCC
plt.subplot(5, 2, 3)
librosa.display.specshow(mfcc_original, sr=original_stopa, x_axis='time')
plt.colorbar()
plt.title('MFCC Originalnog Audio Zapisa')

# Krivotvorenog audio MFCC
plt.subplot(5, 2, 4)
librosa.display.specshow(mfcc_krivotvoren, sr=neovlastena_stopa, x_axis='time')
plt.colorbar()
plt.title('MFCC Krivotvorenog Audio Zapisa')

# Original audio DFT
plt.subplot(5, 2, 5)
plt.plot(dft_original)
plt.title('DFT Originalnog Audio Zapisa')

# Tampered audio DFT
plt.subplot(5, 2, 6)
plt.plot(dft_krivotvoren)
plt.title('DFT Krivotvorenog Audio Zapisa')

# Original audio chromagram
plt.subplot(5, 2, 7)
librosa.display.specshow(chroma_original, sr=original_stopa, x_axis='time')
plt.plot(chroma_original)
plt.title('Chromagram Originalnog Audio Zapisa')

# Krivotvoren audio chromagram
plt.subplot(5, 2, 8)
librosa.display.specshow(chroma_krivotvoren, sr=neovlastena_stopa, x_axis='time')
plt.plot(chroma_krivotvoren)
plt.title('Chromagram krivotvorenog Audio Zapisa')

# Prikaz the original chromagram
plt.subplot(5, 2, 9)
librosa.display.specshow(C_original, sr=original_stopa, x_axis='time', y_axis='chroma')
plt.title('Original')
plt.colorbar()

# Prikaz  krivotvorenog chromagram
plt.subplot(5, 2, 10)
librosa.display.specshow(C_krivotvoren, sr=neovlastena_stopa, x_axis='time', y_axis='chroma')
plt.title('Krivotvoren')
plt.colorbar()

# Prikaz
plt.tight_layout()
plt.show()