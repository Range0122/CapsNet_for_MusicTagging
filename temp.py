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
    model = models.load_model('check_point/PureCapsNet_best.h5')
    model.summary()
