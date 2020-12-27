import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import random


def compute_melspectrogram(audio_path, sr=22050, mels=96):
    SR = sr
    N_FFT = 512
    N_MELS = mels
    HOP_LEN = 256
    DURA = 29.12

    src, sr = librosa.load(audio_path, sr=SR)
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:
        src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]
    logam = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS))
    ret = ret[:, :, np.newaxis]
    return ret


def draw_signal(audio_path):
    y, sr = librosa.load(audio_path)
    fig, ax = plt.subplots()
    librosa.display.waveplot(y=y, sr=sr, ax=ax)

    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.show()


def draw_melspectrogram(audio_path):
    # data augmentation or not
    y, sr = librosa.load(audio_path, sr=None)
    # y, sr = dropout(audio_path, None, 0.05)
    # y, sr = gaussian_noise(audio_path, None, 0.05, 0)
    y, sr = pitch_shifting(audio_path, sr=None, n_steps=6, bins_per_octave=12)
    # y, sr = time_stretching(audio_path, sr=None, rate=2)

    D = np.abs(librosa.stft(y)) ** 2
    S = librosa.feature.melspectrogram(S=D, sr=sr)
    # S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Loudness
    # S_dB = loudness(S_dB, 10)

    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')

    plt.xlabel("Time(s)")
    plt.ylabel("Frequency(Hz)")
    plt.show()


def dropout(audio_path, sr=None, p=0.05):
    # TODO:p = 0.05, 0.10, 0.15, 0.20
    y, sr = librosa.load(audio_path, sr)
    for i in range(len(y)):
        is_dropout = random.randint(0, 100) < 100 * p
        if is_dropout:
            y[i] = 0
    return y, sr


def gaussian_noise(audio_path, sr=None, sigma=0.05, mu=0):
    # TODO: sigma = 0.05, 0.1, 0.2
    y, sr = librosa.load(audio_path, sr)
    noise = sigma * sigma * np.random.randn(y.shape[0],) + mu
    y += noise
    return y, sr


def pitch_shifting(audio_path, sr=None, n_steps=0.0, bins_per_octave=12):
    # TODO:
    y, sr = librosa.load(audio_path, sr)
    y = librosa.effects.pitch_shift(y, sr, n_steps=n_steps, bins_per_octave=bins_per_octave)
    return y, sr


def time_stretching(audio_path, sr=None, rate=1):
    # TODO: rate =
    y, sr = librosa.load(audio_path, sr)
    y = librosa.effects.time_stretch(y, rate)
    return y, sr


def loudness(S_dB, db=10):
    # TODO: db = +-5, +-10, +-20
    ans = S_dB + db
    ans[0][0] = -80
    return ans


if __name__ == "__main__":
    path = '/Users/range/Code/Data/example/1.mp3'
    melspectrogram = compute_melspectrogram(audio_path=path, sr=22050, mels=96)
    print(melspectrogram.shape)
    # draw_signal(audio_path=path)
    # draw_melspectrogram(audio_path=path)
