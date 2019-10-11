from utils import load_all_data
from models import *
import librosa.display
import config as C


def draw_spectrogram(feature):
    feature = np.reshape(feature, (96, 1366))
    librosa.display.specshow(feature, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # path = C.PATH
    path = '/home/range/Data/MusicFeature/MTAT/log_spectrogram/sub_test'

    train_model, eval_model, manipulate_model = CapsNet(input_shape=C.INPUT_SHAPE, n_class=C.OUTPUT_CLASS,
                                                        routings=C.ROUTINGS)
    eval_model.load_weights(f'check_point/{train_model.name}_best.h5')
    x_test, y_test = load_all_data(path, target='test')
    y_pred = eval_model.predict(x_test)

    for i in range(1):
        alpha = x_test[i][0][0] / y_pred[1][i][0][0]
        draw_spectrogram(x_test[i])
        draw_spectrogram(y_pred[1][i])
        # print('='*10 + str(i+1) + '='*10)
        # print(x_test[i][0])
        # print(librosa.power_to_db(x_test[i][0], ref=np.max))
        # print(y_pred[1][i][0])



