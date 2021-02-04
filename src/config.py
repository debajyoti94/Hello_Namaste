''' Here we define the configuration variables which
will be used throughout this project'''


# for the dataset
RAW_DATASET = '../input/hin.txt'
PRETRAINED_EMBEDDINGS = '../glove_embeddings/model.txt'
MAX_SAMPLES = 2000 # considering only 2000 samples for now
SEQUENCE_DELIMITER = '\t'


# for the model
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300
BATCH_SIZE = 128
LR = 0.01
NUM_EPOCHS = 100
LSTM_HIDDEN_VECTORS = 100
LOSS_FN = 'categorical_crossentropy'
VALIDATION_SPLIT = 0.2


# filenames which will be used for storing
# different types of data
INPUT_SEQ_ENCODER = '../input/input_seq_encoder.pickle'
INPUT_SEQ_DECODER = '../input/input_seq_decoder.pickle'
OUTPUT_SEQ_DECODER = '../input/output_seq_target_decoder.pickle'
INPUT_TOKENIZER = '../input/input_tokenizer.pickle'
TRANSLATION_TOKENIZER = '../input/translation_tokenizer.pickle'
ACTUAL_VOCAB_SIZE_INPUT = '../input/actual_vocab_size_input.pickle'
ACTUAL_VOCAB_SIZE_TRANSLATION = '../input/actual_vocab_size_translation.pickle'

PADDED_ENCODER_INPUT_SEQ = '../input/padded_encoder_input_seq.pickle'
PADDED_DECODER_INPUT_SEQ = '../input/padded_decoder_input_seq.pickle'
PADDED_DECODER_OUTPUT_SEQ = '../input/padded_decoder_output_seq.pickle'
MAX_ENCODER_INPUT_SEQ_LEN = '../input/max_encoder_input_seq_len.pickle'
MAX_DECODER_INPUT_SEQ_LEN = '../input/max_decoder_input_seq_len.pickle'

EMBEDDING_MATRIX = '../input/embedding_matrix.pickle'
WORD_VECTOR_DICT = '../input/word_vector_dict.pickle'

# for plots
LOSS_PLOT = '../plots/loss_plot.png'
ACCURACY_PLOT = '../plots/accuracy_plot.png'

# models
TEACHER_FORCING_MODEL = '../models/tf_model.h5'
INFERENCE_ENCODER_MODEL = '../models/inf_encoder_model.h5'
INFERENCE_DECODER_MODEL = '../models/inf_decoder_model.h5'