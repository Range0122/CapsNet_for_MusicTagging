from utils import data_generator, load_all_data, progress
from sklearn import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.metrics import categorical_accuracy
from keras import optimizers
from models import *
import os
import config as C
import tensorflow as tf
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
sess = tf.Session(config=config)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=['train', 'test'], help='train or test')
    return parser.parse_args()


def main(args):
    path = C.PATH

    model, eval_model, manipulate_model = CapsNet(input_shape=C.INPUT_SHAPE, n_class=C.OUTPUT_CLASS,
                                                  routings=C.ROUTINGS)
    model.summary()

    if args.target == 'train':
        model.compile(optimizer=optimizers.Adam(lr=C.LR),
                      loss=[margin_loss, 'mse'],
                      loss_weights=[1., C.LAM_RECON],
                      metrics={'capsnet': 'accuracy'})

        model.fit_generator(data_generator('/'.join((path, 'train')), target='train'), epochs=50,
                            steps_per_epoch=C.TRAIN_SIZE // C.BATCH_SIZE,
                            validation_data=data_generator('/'.join((path, 'val')), target='val'),
                            validation_steps=C.VAL_SIZE // C.BATCH_SIZE, verbose=1,
                            callbacks=[ModelCheckpoint(f'check_point/{model.name}_best.h5',
                                                       monitor='val_capsnet_acc', save_best_only=True,
                                                       save_weights_only=True, verbose=1),
                                       ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                                                         mode='min'),
                                       EarlyStopping(monitor='val_loss', patience=20),
                                       LearningRateScheduler(schedule=lambda epoch: C.LR * (C.LR_DECAY ** epoch))]
                            )
        model.save_weights(f'check_point/{model.name}_final.h5')
    else:
        # manipulate_latent(manipulate_model, (x_test, y_test), args)
        # test(model=eval_model, data=(x_test, y_test), args=args)

        # eval_model.load_weights(f'check_point/{train_model.name}_best.h5')
        eval_model.load_weights(f'check_point/model_1_best.h5')

        x_test, y_test = load_all_data('/'.join((path, 'test')), target='test')

        index = 0
        batch_size = 200
        print("length of y_test:", len(y_test))

        test_input = x_test[index: index + batch_size]
        y_pred = eval_model.predict(test_input)[0]
        index += batch_size

        while index < len(y_test) - batch_size:
            test_input = x_test[index: index + batch_size]
            temp = eval_model.predict(test_input)[0]
            y_pred = np.vstack((y_pred, temp))
            index += batch_size

            percent = index / len(y_test)
            progress(percent, width=30)

        test_input = x_test[index:]
        temp = eval_model.predict(test_input)[0]
        y_pred = np.vstack((y_pred, temp))

        # print(y_test.shape, y_pred.shape)

        # print("START COMPUTING...")
        rocauc = metrics.roc_auc_score(y_test, y_pred)
        prauc = metrics.average_precision_score(y_test, y_pred, average='macro')
        y_pred = (y_pred > 0.5).astype(np.float32)
        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average='samples')

        print(f'Test scores: rocauc={rocauc:.6f}\tprauc={prauc:.6f}\tacc={acc:.6f}\tf1={f1:.6f}')


if __name__ == "__main__":
    args = get_arguments()
    main(args)
