import librosa
import numpy as np
import sys
import random
import pandas as pd


def compute_melgram(audio_path):
    SR = 22050
    N_FFT = 2048
    N_MELS = 128
    HOP_LEN = 256
    DURA = 29.09

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
    df = pd.read_csv('/home/range/Data/MTAT/raw/annotations_final.csv', delimiter='\t')
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
    audio_root = '/home/range/Data/MTAT/raw/mp3/'
    npy_root = '/home/range/Data/MusicFeature/MTAT/log_spectrogram2'
    for i in range(len(dataset[0])):
        try:
            path = ''.join((audio_root, dataset[0][i]))
            # TODO: CHANGE FEATURE GENERATOR HERE.
            feature = compute_melgram(path)

            file = dataset[0][i][2:-4]
            npy_path = '/'.join((npy_root, set_type, file))
            np.save(npy_path, feature)

            i += 1
            percent = i / len(dataset[0])
            progress(percent, width=30)

        except Exception as e:
            print(e)


def progress(percent, width=50):
    if percent > 1:  # 如果百分比大于1的话则取1
        percent = 1
    show_str = ('[%%-%ds]' % width) % (int(percent * width) * '#')
    # 一共50个#，%d 无符号整型数,-代表左对齐，不换行输出，两个% % 代表一个单纯的%，对应的是后面的s，后面为控制#号的个数
    # print(show_str)  #[###############               ] show_str ，每次都输出一次
    print('\r%s %s%%' % (show_str, int(percent * 100)), end='', file=sys.stdout, flush=True)
    # \r 代表调到行首的意思，\n为换行的意思，fiel代表输出到哪，flush=True代表无延迟，立马刷新。第二个%s是百分比


if __name__ == '__main__':
    test_path = '/home/range/Data/MTAT/raw/mp3/2/zephyrus-angelus-10-ave_maria___benedicta_to_ockeghem-59-88.mp3'
    feature = compute_melgram(test_path)
    print(feature.shape)

    # train, val, test = create_dataset_for_MTAT()
    # generate_feature_for_MTAT(train, 'train')
    # generate_feature_for_MTAT(val, 'val')
    # generate_feature_for_MTAT(test, 'test')
