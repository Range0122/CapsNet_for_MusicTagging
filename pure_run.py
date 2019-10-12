from utils import data_generator, load_all_data, batch_prediction, model_evaluate
from keras import callbacks
from keras.metrics import categorical_accuracy
from models import *
import os
import config as C
import tensorflow as tf
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.90
sess = tf.Session(config=config)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=['train', 'test'], help='train or test')
    return parser.parse_args()


def main(args):
    path = C.PATH
    model = PureCapsNet(input_shape=C.INPUT_SHAPE, n_class=C.OUTPUT_CLASS, routings=C.ROUTINGS)
    model.summary()
    # exit()

    if args.target == 'train':
        checkpoint = callbacks.ModelCheckpoint(f'check_point/{model.name}_best.h5', monitor='val_loss',
                                               save_best_only=True, verbose=1)
        reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='min')
        earlystopping = callbacks.EarlyStopping(monitor='val_loss', patience=20)
        log = callbacks.CSVLogger('logs/log.csv')
        tb = callbacks.TensorBoard('logs/tensorboard-logs', batch_size=C.BATCH_SIZE, histogram_freq=0)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: C.LR * (C.LR_DECAY ** epoch))

        model.compile(optimizer=optimizers.Adam(lr=C.LR),
                      loss=[margin_loss],
                      loss_weights=[1.],
                      # metrics=[categorical_accuracy],
                      metrics={'capsnet': 'accuracy'})

        model.fit_generator(data_generator('/'.join((path, 'train')), target='train'), epochs=20,
                            steps_per_epoch=C.TRAIN_SIZE // C.BATCH_SIZE,
                            validation_data=data_generator('/'.join((path, 'val')), target='val'),
                            validation_steps=C.VAL_SIZE // C.BATCH_SIZE, verbose=1,
                            callbacks=[checkpoint, reduce, log, tb, earlystopping, lr_decay])
        model.save(f'check_point/{model.name}_final.h5')
    else:
        # model.load_weights(f'check_point/{model.name}_best.h5')
        model.load_weights(f'check_point/{model.name}_final.h5')
        print("Loading test data ...")
        x_test, y_test = load_all_data('/'.join((path, 'test')), target='test')
        y_pred = batch_prediction(model, x_test, batch_size=200)
        model_evaluate(y_pred, y_test)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
