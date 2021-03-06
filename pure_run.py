from utils import data_generator, load_all_data, load_all_data2, batch_prediction, model_evaluate
from keras import callbacks
from keras.metrics import categorical_accuracy
from models import *
import os
import config as C
import tensorflow as tf
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
sess = tf.Session(config=config)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=['train', 'test', 'retrain'], help='train or test')
    return parser.parse_args()


def main(args):
    path = C.PATH
    # model = PureCapsNet(input_shape=C.INPUT_SHAPE, n_class=C.OUTPUT_CLASS, routings=C.ROUTINGS)
    model = TestMixCapsNet(input_shape=C.INPUT_SHAPE, n_class=C.OUTPUT_CLASS, routings=C.ROUTINGS)
    # model = MultiScaleCapsNet(input_shape=C.INPUT_SHAPE, n_class=C.OUTPUT_CLASS, routings=C.ROUTINGS)
    model.summary()
    # exit()

    if args.target == 'train' or args.target == 'retrain':
        checkpoint = callbacks.ModelCheckpoint(f'check_point/{model.name}_best.h5', monitor='val_loss',
                                               save_best_only=True, verbose=1)
        reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='min')
        earlystopping = callbacks.EarlyStopping(monitor='val_loss', patience=20)
        log = callbacks.CSVLogger('logs/log.csv')
        tb = callbacks.TensorBoard('logs/tensorboard-logs', batch_size=C.BATCH_SIZE, histogram_freq=0)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: C.LR * (C.LR_DECAY ** epoch))

        if args.target == 'retrain':
            # sgd with lr=0.01 for fine-tune
            optimizer = optimizers.sgd(lr=0.01, momentum=0.9, nesterov=True, decay=1e-6)
            model.load_weights(f'check_point/{model.name}_best.h5', by_name=True)
            print(f"{model.name} loaded.")
        else:
            optimizer = optimizers.Adam(lr=C.LR)
            print("No model loaded.")

        model.compile(optimizer=optimizer,
                      # loss=[margin_loss],
                      loss='binary_crossentropy',
                      # loss_weights=[1.],
                      metrics=[categorical_accuracy])
        # metrics={'capsnet': 'accuracy'})
        model.fit_generator(data_generator('/'.join((path, 'train'))), epochs=120,
                            steps_per_epoch=C.TRAIN_SIZE // C.BATCH_SIZE,
                            validation_data=data_generator('/'.join((path, 'val'))),
                            validation_steps=C.VAL_SIZE // C.BATCH_SIZE, verbose=1,
                            callbacks=[checkpoint, log, tb, earlystopping])
        # callbacks=[checkpoint])
        model.save(f'check_point/{model.name}_final.h5')
    else:
        # model.load_weights(f'check_point/{model.name}_best.h5')
        model.load_weights(f'check_point/{model.name}_0.904204.h5')

        print("Loading test data ...")
        x_test, y_test = load_all_data('/'.join((path, 'test')))
        y_pred = batch_prediction(model, x_test, batch_size=200)

        print(len(y_test), len(y_pred), len(y_test))
        model_evaluate(y_pred, y_test)

        # model.load_weights(f'check_point/{model.name}_final.h5')
        # y_pred = batch_prediction(model, x_test, batch_size=200)
        # model_evaluate(y_pred, y_test)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
