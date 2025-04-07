import os
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
np.seterr(all="ignore")

import librosa
from librosa.util import normalize
import torch

import parselmouth
from parselmouth.praat import call
from scipy import signal, interpolate

def librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax):
    return librosa.filters.mel(
        sr=sampling_rate,
        n_fft=n_fft,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax
    )

def trim_silence(wav, top_db, fft_size, hop_size):
    '''Trim leading and trailing silence
    Useful for M-AILABS datasets if we choose to trim the extra 0.5 silence at beginning and end.
    '''
    return librosa.effects.trim(wav, top_db=top_db, frame_length=fft_size, hop_length=hop_size)[0]

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_compression_torch(x, clip_val=1e-5, C=1):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes, clip_val, C=1):
    output = dynamic_range_compression_torch(magnitudes, clip_val, C)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, eps, clip_val, center=False):
    global mel_basis, hann_window

    key = str(fmax) + '_' + str(y.device)
    if key not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode='reflect'
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=hann_window[str(y.device)],
        center=center, pad_mode='reflect',
        normalized=False, onesided=True
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + eps)
    spec = torch.matmul(mel_basis[key], spec)
    spec = spectral_normalize_torch(spec, clip_val)

    return spec

def call_generate_melspec(audio_org, sampling_rate, MAX_WAV_VALUE, n_fft, num_mels, hop_size, win_size, fmin, fmax, eps, clip_val):
    audio = audio_org / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95
    audio = trim_silence(audio, top_db=20, fft_size=win_size, hop_size=hop_size)
    audio = torch.FloatTensor(audio).unsqueeze(0)

    mel = mel_spectrogram(audio, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, eps, clip_val, center=False)

    return mel.squeeze(0).transpose(0, 1), audio.squeeze(0)

def stretch_time_series(feature, mel_len):
    feature_min = np.min(feature[feature > 0])

    f = interpolate.interp1d(
        np.arange(0, len(feature)),
        feature
    )
    feature = f(
        np.linspace(0, len(feature) - 1, mel_len)
    )

    feature[feature < feature_min] = 0.0

    return feature

def generate_features(audio, melspec, sampling_rate=16000, hopsize=256, winsize=1024):
    mel_len = melspec.shape[0]
    snd = parselmouth.Sound(audio, sampling_frequency=sampling_rate)

    pitch = snd.to_pitch()
    pitch = pitch.selected_array['frequency']

    pitch = stretch_time_series(pitch, mel_len)

    return pitch

def fs(audio, sampling_rate=16000):
    sound = parselmouth.Sound(audio, sampling_frequency=sampling_rate)

    pitch_ratio = np.random.uniform(0.6, 1.4, size=1)[0]
    choose_flag = np.random.uniform(0, 1.0, size=1)[0]

    if choose_flag > 0.5:
        manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
        pitch_tier = call(manipulation, "Extract pitch tier")
        call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, pitch_ratio)
        call([pitch_tier, manipulation], "Replace pitch tier")
        data = call(manipulation, "Get resynthesis (overlap-add)")
        data = data.values.reshape(-1)
    else:
        data = audio

    return data
