### Modules needed: numpy, librosa, soundfile, matplotlib, scipy

import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.fftpack as fft
from scipy.signal import medfilt

### You will need to rename the next line file name to use your own audio file.
y, sr = librosa.load('input1.wav', sr=None)

S_full, phase = librosa.magphase(librosa.stft(y))
noise_power = np.mean(S_full[:, :int(sr*0.0008)], axis=1)
mask = S_full > noise_power[:, None]
mask = mask.astype(float)
mask = medfilt(mask, kernel_size=(1,5))
S_clean = S_full * mask
y_clean = librosa.istft(S_clean * phase)

### Saving the cleaned audio file as 'clean.wav'.
sf.write('clean3.wav', y_clean, sr)