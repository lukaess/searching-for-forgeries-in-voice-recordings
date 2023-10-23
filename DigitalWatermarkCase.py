import numpy as np
import scipy.io.wavfile as wav
from scipy import signal

# ucitavanje audio zapisa
stopa, audio = wav.read('SplicingOrgExample.wav')


# kreiranje jednostavnog vodenog žiga tako što dodamo sinusni val na odredjenoj frekvenciji
watermark_freq = 1000  # 1 kHz

watermark = np.sin(2 * np.pi * watermark_freq * np.arange(len(audio)) / stopa)

# Dodavanje vodenog žiga u audio zapis
watermarked_audio = audio + watermark


# Spremanje u bazu audio zapisa sa vodenim žigom
wav.write('watermarked.wav', stopa, watermarked_audio)


# simulirajmo scenarij u kojem primamo audio datoteku i želimo provjeriti je li uredjena
# Pretpostavit cemo da će svako uredjivanje ukloniti ili izmijeniti vodeni žig


# Ucitavanje potencijalno manipuliranog audio zapisa
stopa_manipulacije, manipuliran_audio = wav.read('CopyMoveForgery.wav')


# Racunanje unakrsne korelacije izmedju uredjenog zvuka i vodenog žiga
cross_correlation = signal.correlate(manipuliran_audio, watermark, mode='same')


# Ako zvuk nije uredjivan, unakrsna korelacija trebala bi imati vrhunac na frekvenciji vodenog žiga

# Koristit ćemo FFT da pronadjemo frekvenciju s najvećom magnitudom u unakrsnoj korelaciji
freqs = np.fft.rfftfreq(len(cross_correlation), 1 / stopa)
fft = np.abs(np.fft.rfft(cross_correlation))


# Pronađite frekvenciju s maksimalnom veličinom u FFT-u
max_freq = freqs[np.argmax(fft)]


# Ako je maksimalna frekvencija jednaka frekvenciji vodenog žiga, zvuk nije uredjivan
if np.isclose(max_freq, watermark_freq):

    print("Audio zapis nije manipuliran.")

else:

    print("Audio zapis je manipuliran.")