import os
import pandas as pd
import numpy as np
import config as C
import random
import math
import sys
from sklearn import metrics
import matplotlib.pyplot as plt


def plot_log(filename, show=True):
    data = pd.read_csv(filename)

    fig = plt.figure(figsize=(4, 6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num) / width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num) / width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num) / height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[:, :, 0]
    return image


def shuffle_both(a, b):
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(a)
    random.seed(randnum)
    random.shuffle(b)
    return a, b


def progress(percent, width=50):
    if percent > 1:  # 如果百分比大于1的话则取1
        percent = 1
    show_str = ('[%%-%ds]' % width) % (int(percent * width) * '#')
    # 一共50个#，%d 无符号整型数,-代表左对齐，不换行输出，两个% % 代表一个单纯的%，对应的是后面的s，后面为控制#号的个数
    # print(show_str)  #[###############               ] show_str ，每次都输出一次
    print('\r%s %s%%' % (show_str, int(percent * 100)), end='', file=sys.stdout, flush=True)
    # \r 代表调到行首的意思，\n为换行的意思，fiel代表输出到哪，flush=True代表无延迟，立马刷新。第二个%s是百分比


def data_generator(path, target='default'):
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
    # tags = ['classical', 'techno', 'electronic', 'rock', 'pop', 'country', 'metal', 'jazz', 'modern', 'baroque',
    # 'folk', 'punk']

    df = pd.read_csv('/home/range/Data/MTAT/raw/annotations_final.csv', delimiter='\t')
    mp3_paths = list(df['mp3_path'].values)
    labels = df[tags].values

    for i in range(len(mp3_paths)):
        mp3_paths[i] = mp3_paths[i].split('/')[-1][:-4]

    for root, dirs, files in os.walk(path):
        length = len(files)
        np.random.shuffle(files)
        while True:
            batch_size = C.BATCH_SIZE
            index = 0
            while index < length - batch_size:
                x = []
                y = []

                for file in files[index:index + batch_size]:
                    file_path = '/'.join((root, file))

                    feature = np.load(file_path)
                    label = labels[mp3_paths.index(file[:-C.LENGTH])]
                    if label.sum() != 0 and label.sum() != 1:
                        if target == 'short':
                            start = 0
                            while start + 96 < 1366:
                                x.append(feature[:, start: start + 96, :])
                                y.append(label)
                                start += 48

                        else:
                            x.append(feature)
                            y.append(label)

                index += batch_size

                x, y = shuffle_both(x, y)

                yield [np.array(x), np.array(y)]

                # if target == 'test':
                #     yield np.array(x)
                # else:
                # yield [np.array(x), np.array(y)], [np.array(y), np.array(x)]
                # yield [np.array(x), np.array(y)], [np.array(x)]
                # yield np.array(x), np.array(y)


def load_all_data(path, target='default'):
    """
    path为存放所有的npy文件的目录
    Used for MTAT Dataset
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
    # tags = ['classical', 'techno', 'electronic', 'rock', 'pop', 'country', 'metal', 'jazz', 'modern', 'baroque',
    # 'folk', 'punk']

    df = pd.read_csv('/home/range/Data/MTAT/raw/annotations_final.csv', delimiter='\t')
    mp3_paths = list(df['mp3_path'].values)
    labels = df[tags].values

    for i in range(len(mp3_paths)):
        mp3_paths[i] = mp3_paths[i].split('/')[-1][:-4]

    x = []
    y = []
    for root, dirs, files in os.walk(path):
        i = 0
        for file in files:
            file_path = '/'.join((root, file))

            feature = np.load(file_path)
            label = labels[mp3_paths.index(file[:-C.LENGTH])]

            if label.sum() != 0 and label.sum() != 1:
                if target == 'short':
                    start = 0
                    while start + 96 < 1366:
                        x.append(feature[:, start: start + 96, :])
                        y.append(label)
                        start += 48

                else:
                    x.append(feature)
                    y.append(label)
                    # y.append(np.where(label == 1)[0])

            i += 1
            percent = i / len(files)
            progress(percent, width=30)
        print('\n')

    x, y = shuffle_both(x, y)

    # if target == 'test':
    #     return np.array(x), np.array(y)
    # else:
    #     return [np.array(x), np.array(y)], [np.array(y), np.array(x)]
    return np.array(x), np.array(y)


def batch_prediction(model, x_test, batch_size):
    index = 0
    batch_size = batch_size
    print("length of x_test:", len(x_test))

    print("Predicting y_pred ...")
    batch_input = x_test[index: index + batch_size]
    y_pred = model.predict(batch_input)
    index += batch_size
    percent = index / len(x_test)
    progress(percent, width=30)

    while index < len(x_test) - batch_size:
        batch_input = x_test[index: index + batch_size]
        batch_output = model.predict(batch_input)
        y_pred = np.vstack((y_pred, batch_output))
        index += batch_size

        percent = index / len(x_test)
        progress(percent, width=30)

    test_input = x_test[index:]
    temp = model.predict(test_input)
    y_pred = np.vstack((y_pred, temp))
    percent = index / len(x_test)
    progress(percent, width=30)

    return y_pred


def model_evaluate(y_pred, y_true):
    print("Evaluating model ... \n")
    rocauc = metrics.roc_auc_score(y_true, y_pred)
    prauc = metrics.average_precision_score(y_true, y_pred, average='macro')
    y_pred = (y_pred > 0.5).astype(np.float32)
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='samples')

    print(f'Test scores: rocauc={rocauc:.6f}\tprauc={prauc:.6f}\tacc={acc:.6f}\tf1={f1:.6f}')


def load_all_data2(path):
    """
    path为存放所有的npy文件的目录
    Used for GTZAN log_spectrogram
    """
    tags = ['blues', 'classical', 'disco', 'country', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    x = []
    y = []

    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = '/'.join((path, file))
            feature = np.load(file_path)
            label = tags.index(file.split('.')[0])

            x.append(feature)
            y.append(label)

    x, y = shuffle_both(x, y)

    return np.array(x), np.array(y)


if __name__ == "__main__":
    # path = '/home/range/Data/MusicFeature/MTAT/short_spectrogram'
    # x_val, y_val = load_all_data('/'.join((path, 'val')))
    # path = '/home/range/Data/MusicFeature/MTAT/short_spectrogram/val/' \
    #        'william_brooks-blue_ribbon__the_best_of_william_brooks-11-grace-0-2900.npy'
    # feature = np.load(path)
    # print(feature.shape)

    # path = '/home/range/Data/MusicFeature/GTZAN/log_spectrogram/train'
    # load_all_data2(path)
    g = data_generator('/'.join((C.PATH, 'train')), target='short')
    x, y = next(g)
    print(x.shape)
    print(y.shape)
