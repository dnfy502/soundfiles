import numpy as np
import librosa
import soundfile as sf

# Parameters
input_file = 'input1.wav'
output_file = 'output_audio_denoised.wav'
segment_duration = 60  # seconds
noise_fraction = 0.0003  # first 10% for noise estimation

# Load audio file
y, sr = librosa.load(input_file, sr=None)
S_full = librosa.stft(y)
total_duration = librosa.get_duration(y=y, sr=sr)
num_segments = int(total_duration // segment_duration)
samples_per_segment = int(sr * segment_duration)

noise_powers = []

for i in range(num_segments):
    start_idx = i * samples_per_segment
    end_idx = start_idx + samples_per_segment
    segment = S_full[:, start_idx:end_idx]
    noise_power = np.mean(segment[:, :int(noise_fraction * samples_per_segment)], axis=1)
    noise_powers.append(noise_power)

# Handle any remaining samples
if end_idx < S_full.shape[1]:
    segment = S_full[:, end_idx:]
    noise_power = np.mean(segment[:, :int(noise_fraction * segment.shape[1])], axis=1)
    noise_powers.append(noise_power)

# Convert list to numpy array for aggregation
noise_powers = np.array(noise_powers)

# Aggregate noise power using median
aggregated_noise_power = np.median(noise_powers, axis=0)

# Apply noise reduction
S_denoised = S_full - aggregated_noise_power[:, np.newaxis]
S_denoised = np.maximum(S_denoised, 0)  # Ensure no negative values

# Reconstruct the audio signal
y_denoised = librosa.istft(S_denoised)

# Save the denoised audio
sf.write(output_file, y_denoised, sr)