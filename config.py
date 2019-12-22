# SIZE
# TRAIN_SIZE = 6000
# VAL_SIZE = 2000
# TRAIN_SIZE = 280485
# VAL_SIZE = 27270
BATCH_SIZE = 8

TRAIN_SIZE = 18706
VAL_SIZE = 1825

# TRAIN
LR = 0.001
INPUT_SHAPE = (96, 1366, 1)
# INPUT_SHAPE = (96, 96, 1)
OUTPUT_CLASS = 50

# for changing the input shape
LENGTH = 4
PATH = '/home/range/Data/MusicFeature/MTAT/log_spectrogram'
# PATH = '/home/range/Data/MusicFeature/GTZAN/log_spectrogram'

# "The coefficient for the loss of decoder"
# 0.0005 * 96 *1366 = 655.68
LAM_RECON = 0.358

# "The value multiplied by lr at each epoch. Set a larger value for larger epochs"
LR_DECAY = 0.9

# "The number of iterations used in routing algorithm, which should > 0"
ROUTINGS = 3

DIM_CAPSULE = 8