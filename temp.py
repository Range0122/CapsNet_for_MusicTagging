import librosa
import numpy as np
import math
import os
import sys
import random
import pandas as pd
from python_speech_features import logfbank, fbank


def compute_melgram(audio_path):
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]
    logam = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS))
    ret = ret[:, :, np.newaxis]
    return ret


def shuffle_both(a, b):
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(a)
    random.seed(randnum)
    random.shuffle(b)
    return a, b


def compute_total_feature(path):
    y, sr = librosa.load(path, sr=None, duration=29.12)
    feature = librosa.feature.melspectrogram(y, sr)
    # print(feature.shape)
    # feature = logfbank(y, sr).T

    return feature[np.newaxis, :]


def compute_short_feature(path):
    y, sr = librosa.load(path, sr=None, duration=29.18)

    temp = logfbank(y, sr).T[np.newaxis, :]
    # spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=39)[np.newaxis, :]
    # temp = librosa.amplitude_to_db(spectrogram ** 2, ref=1.0)

    feature = []
    overlap = 0.5
    feature_length = 320
    audio_length = temp.shape[2]
    feature_num = int(math.floor(((audio_length / feature_length) - 1) / (1 - overlap) + 1))

    for i in range(feature_num):
        start = int(i * (1 - overlap) * feature_length)
        end = int((1 + i * (1 - overlap)) * feature_length)
        feature.append(temp[:, :, start: end])

    return feature


def generate_data(path):
    """
    path为存放所有的npy文件的目录
    """
    x = []
    y = []
    labels = ['hiphop', 'disco', 'country', 'classical', 'blues', 'reggae', 'rock', 'jazz', 'metal', 'pop']
    for root, dirs, files in os.walk(path):
        # file_list = [root + '/' + file for file in files]
        file_list = []
        for file in files:
            file_list += '/'.join((root, file))

        i = 0
        for file in file_list:
            label = file.split('/')[-1].split('.')[0]
            label = labels.index(label)
            feature = np.load(file)

            x.append(feature)
            # x.append(np.mean(feature, axis=2).flatten())
            # x.append(feature.flatten())
            y.append(label)

            i += 1
            percent = i / len(file_list)
            progress(percent, width=30)

    x, y = shuffle_both(x, y)

    return np.array(x), np.array(y)


def data_generator_for_MTAT(path):
    """
    path为存放所有的npy文件的目录
    """
    # load annotation csv
    tags = ['choral', 'female voice', 'metal', 'country', 'weird', 'no voice',
            'cello', 'harp', 'beats', 'female vocal', 'male voice', 'dance',
            'new age', 'voice', 'choir', 'classic', 'man', 'solo', 'sitar', 'soft',
            'pop', 'no vocal', 'male vocal', 'woman', 'flute', 'quiet', 'loud',
            'harpsichord', 'no vocals', 'vocals', 'singing', 'male', 'opera',
            'indian', 'female', 'synth', 'vocal', 'violin', 'beat', 'ambient',
            'piano', 'fast', 'rock', 'electronic', 'drums', 'strings', 'techno',
            'slow', 'classical', 'guitar']

    df = pd.read_csv('/home/range/Data/MTAT/raw/annotations_final.csv', delimiter='\t')
    mp3_paths = list(df['mp3_path'].values)
    labels = df[tags].values

    for i in range(len(mp3_paths)):
        mp3_paths[i] = mp3_paths[i].split('/')[-1][:-4]

    for root, dirs, files in os.walk(path):
        length = len(files)
        np.random.shuffle(files)
        batch_size = 128
        index = 0
        while index < length - batch_size:
            x = []
            y = []

            for file in files[index:index + batch_size]:
                file_path = '/'.join((root, file))

                feature = np.load(file_path)
                label = labels[mp3_paths.index(file[:-4])]

                x.append(feature)
                y.append(label)

            index += batch_size

            x, y = shuffle_both(x, y)
            yield np.array(x), np.array(y)


def generate_data_from_MTAT(path):
    """
    path为存放所有的npy文件的目录
    """
    # load annotation csv
    tags = ['choral', 'female voice', 'metal', 'country', 'weird', 'no voice',
            'cello', 'harp', 'beats', 'female vocal', 'male voice', 'dance',
            'new age', 'voice', 'choir', 'classic', 'man', 'solo', 'sitar', 'soft',
            'pop', 'no vocal', 'male vocal', 'woman', 'flute', 'quiet', 'loud',
            'harpsichord', 'no vocals', 'vocals', 'singing', 'male', 'opera',
            'indian', 'female', 'synth', 'vocal', 'violin', 'beat', 'ambient',
            'piano', 'fast', 'rock', 'electronic', 'drums', 'strings', 'techno',
            'slow', 'classical', 'guitar']
    df = pd.read_csv('/home/range/Data/MTAT/raw/annotations_final.csv', delimiter='\t')
    mp3_paths = list(df['mp3_path'].values)
    labels = df[tags].values

    for i in range(len(mp3_paths)):
        mp3_paths[i] = mp3_paths[i].split('/')[-1][:-4]

    x = []
    y = []
    for root, dirs, files in os.walk(path):
        # file_list = [root + '/' + file for file in files]

        i = 0
        for file in files:
            file_path = '/'.join((root, file))

            feature = np.load(file_path)
            label = labels[mp3_paths.index(file[:-4])]

            x.append(feature)
            y.append(label)

            i += 1
            percent = i / len(files)
            progress(percent, width=30)

    x, y = shuffle_both(x, y)

    return np.array(x), np.array(y)


def generate_total_feature(root_path='/home/range/Data/GTZAN/data/'):
    """
    generate feature from GTZAN dataset
    """
    feature_type = 'logfbank'
    labels = ['hiphop', 'disco', 'country', 'classical', 'blues', 'reggae', 'rock', 'jazz', 'metal', 'pop']

    data = {'train': [], 'val': [], 'test': []}

    for label in labels:
        path = root_path + label
        for root, dirs, files in os.walk(path):
            file_list = [root + '/' + file for file in files]
            data['train'] += file_list[:90]
            data['val'] += file_list[90:]
            data['test'] += file_list[90:]

    for item in data.keys():
        i = 0
        for path in data[item]:
            feature = compute_total_feature(path)
            npy_path = f"/home/range/Data/MusicFeature/GTZAN/{feature_type}/{item}/{path.split('/')[-1][:-3]}"
            np.save(npy_path, feature)

            i += 1
            percent = i / len(data[item])
            progress(percent, width=30)


def generate_short_feature(root_path='/home/range/Data/GTZAN/data/'):
    """
    generate feature from GTZAN dataset
    """
    feature_type = 'short_logfbank'
    labels = ['hiphop', 'disco', 'country', 'classical', 'blues', 'reggae', 'rock', 'jazz', 'metal', 'pop']

    data = {'train': [], 'val': [], 'test': []}

    for label in labels:
        path = root_path + label
        for root, dirs, files in os.walk(path):
            file_list = [root + '/' + file for file in files]
            data['train'] += file_list[:80]
            data['val'] += file_list[80:85]
            data['test'] += file_list[85:]

    for item in data.keys():
        i = 0
        for path in data[item]:
            feature_list = compute_short_feature(path)
            feature_num = len(feature_list)
            j = 0
            # # 取所有的feature
            # for feature in feature_list:
            #     npy_path = f"/home/range/Data/MusicFeature/GTZAN/{feature_type}/{item}/{path.split('/')[-1][:-3]}{j}"
            #     np.save(npy_path, feature)
            #     j += 1
            #     i += 1
            #     percent = i/(len(data[item]) * feature_num)
            #     progress(percent, width=30)
            # 只取中间部分的feature，去掉开头和末尾
            for k in range(feature_num - 4):
                k = k + 2
                npy_path = f"/home/range/Data/MusicFeature/GTZAN/{feature_type}/{item}/{path.split('/')[-1][:-3]}{j}"
                feature = feature_list[k]
                np.save(npy_path, feature)
                j += 1
                i += 1
                percent = i / (len(data[item]) * feature_num)
                progress(percent, width=30)


def create_dataset_for_MTAT():
    # load annotation csv
    tags = ['choral', 'female voice', 'metal', 'country', 'weird', 'no voice',
            'cello', 'harp', 'beats', 'female vocal', 'male voice', 'dance',
            'new age', 'voice', 'choir', 'classic', 'man', 'solo', 'sitar', 'soft',
            'pop', 'no vocal', 'male vocal', 'woman', 'flute', 'quiet', 'loud',
            'harpsichord', 'no vocals', 'vocals', 'singing', 'male', 'opera',
            'indian', 'female', 'synth', 'vocal', 'violin', 'beat', 'ambient',
            'piano', 'fast', 'rock', 'electronic', 'drums', 'strings', 'techno',
            'slow', 'classical', 'guitar']
    df = pd.read_csv('/home/Amin/Document/annotations_final.csv', delimiter='\t')
    # mp3_paths = df['mp3_path'].values
    labels = df[tags].values

    # split dataset
    train_paths, val_paths, test_paths = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for index, x in enumerate(df['mp3_path']):
        directory = x.split('/')[0]
        part = int(directory, 16)
        if part in range(12):
            train_paths.append(x)
            train_labels.append(labels[index])
        elif part is 12:
            val_paths.append(x)
            val_labels.append(labels[index])
        elif part in range(13, 16):
            test_paths.append(x)
            test_labels.append(labels[index])

    train_dataset = (train_paths, train_labels)
    val_dataset = (val_paths, val_labels)
    test_dataset = (test_paths, test_labels)

    return train_dataset, val_dataset, test_dataset


def generate_feature_for_MTAT(dataset, set_type):
    """
    The input parameter dataset is from create_dataset_for_MTAT
    eg. (train_paths, train_labels)
    set_type is for train/val/test
    """
    audio_root = '/home/Amin/Document/mp3/'
    npy_root = '/home/Amin/Document/spectrogram'
    for i in range(len(dataset[0])):
        path = ''.join((audio_root, dataset[0][i]))
        feature = compute_melgram(path)

        file = dataset[0][i][2:-4]
        npy_path = '/'.join((npy_root, set_type, file))
        np.save(npy_path, feature)

        i += 1
        percent = i / len(dataset[0])
        progress(percent, width=30)


def progress(percent, width=50):
    if percent > 1:  # 如果百分比大于1的话则取1
        percent = 1
    show_str = ('[%%-%ds]' % width) % (int(percent * width) * '#')
    # 一共50个#，%d 无符号整型数,-代表左对齐，不换行输出，两个% % 代表一个单纯的%，对应的是后面的s，后面为控制#号的个数
    # print(show_str)  #[###############               ] show_str ，每次都输出一次
    print('\r%s %s%%' % (show_str, int(percent * 100)), end='', file=sys.stdout, flush=True)
    # \r 代表调到行首的意思，\n为换行的意思，fiel代表输出到哪，flush=True代表无延迟，立马刷新。第二个%s是百分比


def test(audio_path):
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]
    logam = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS))
    ret = ret[:, :, np.newaxis]
    return ret


if __name__ == '__main__':
    # train, val, test = create_dataset_for_MTAT()
    # generate_feature_for_MTAT(train, 'train')
    # generate_feature_for_MTAT(val, 'val')
    # generate_feature_for_MTAT(test, 'test')
    test_path = '/Users/range/Code/Data/example/0.mp3'
    test_spectrogram = test(test_path)
    print(test_spectrogram.shape)
