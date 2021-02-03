''' Here we define the configuration variables which
will be used throughout this project'''


# for the dataset
RAW_DATASET = '../input/hin.txt'
PRETRAINED_EMBEDDINGS = '../glove_embeddings/model.txt'
MAX_SAMPLES = 2000 # considering only 2000 samples for now
SEQUENCE_DELIMITER = '\t'

INPUT_SEQ_ENCODER = '../input/input_seq_encoder.pickle'
INPUT_SEQ_DECODER = '../input/input_seq_decoder.pickle'
OUTPUT_SEQ_TARGET_DECODER = '../input/output_seq_target_decoder.pickle'

# for the model
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 100
BATCH_SIZE = 128
LR = 0.01
NUM_EPOCHS = 100


# filenames which will be used for storing
# different types of data
EMBEDDING_MATRIX = '../input/embedding_matrix.pickle'
