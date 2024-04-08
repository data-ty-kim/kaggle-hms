class config:
    AMP = True
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VALID = 32
    EPOCHS = 4
    FOLDS = 5
    FREEZE = False
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1e7
    MODEL = "tf_efficientnet_b2"
    NUM_FROZEN_LAYERS = 39
    NUM_WORKERS = 8 # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    TRAIN_FULL_DATA = False
    VISUALIZE = True
    WEIGHT_DECAY = 0.01
    
    
class paths:
    OUTPUT_DIR = "./"
    PRE_LOADED_EEGS = './data/processed/eeg_specs.npy'
    PRE_LOADED_SPECTOGRAMS = './data/download/specs.npy'
    TRAIN_CSV = "./data/raw/train.csv"
    TRAIN_EEGS = "./data/raw/train_eegs/"
    TRAIN_SPECTOGRAMS = "./data/raw/train_spectrograms/"
    