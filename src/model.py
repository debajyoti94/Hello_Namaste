''' Here we create the sequence to sequence model'''

# import modules here
import config

from keras.layers import Embedding, Input, LSTM, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import numpy as np
from matplotlib import pyplot as plt

class TranslationModel:

    def train_translation_model(self, padded_encoder_input_seq,
                                padded_decoder_input_seq,
                                padded_decoder_output_seq,
                                embedding_matrix,
                                actual_vocab_size_input_encoder,
                                actual_vocab_size_input_decoder,
                                max_input_seq_encoder_len,
                                max_input_seq_decoder_len):

        # create the embedding layer
        embedding_layer = Embedding(actual_vocab_size_input_encoder,
                                    config.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=max_input_seq_encoder_len)

        # create one hot targets
        one_hot_targets = np.zeros(shape=(len(padded_decoder_output_seq),
                                          max_input_seq_decoder_len,
                                          actual_vocab_size_input_decoder))

        for seq_index, sequence in enumerate(padded_decoder_output_seq):
            for word_index, word in enumerate(sequence):
                    one_hot_targets[seq_index, word_index, word] = 1


        # create first of two models
        # in the first model we build the
        # encoder and decoder with teacher forcing
        encoder_input_layer = Input(shape=(max_input_seq_encoder_len,))
        x = embedding_layer(encoder_input_layer)
        encoder_lstm = LSTM(config.LSTM_HIDDEN_VECTORS, return_state=True)
        encoder_outputs, h, c = encoder_lstm(x)

        final_encoder_states = [h, c]

        # create decoder model
        decoder_embedding_layer = Embedding(actual_vocab_size_input_decoder,
                                                         config.EMBEDDING_DIM)
        decoder_input_layer = Input(shape=(max_input_seq_decoder_len,))
        decoder_x = decoder_embedding_layer(decoder_input_layer)
        decoder_lstm = LSTM(units=config.LSTM_HIDDEN_VECTORS,
                            return_state=True,
                            return_sequences=True)

        decoder_outputs, _, _ = decoder_lstm(decoder_x,
                                             initial_state=final_encoder_states)

        # final dense layer with softmax activation
        decoder_dense_layer = Dense(actual_vocab_size_input_decoder,
                                    activation='softmax')
        decoder_outputs = decoder_dense_layer(decoder_outputs)

        # teacher forcing model
        tf_model = Model([encoder_input_layer, decoder_input_layer],
                         decoder_outputs)

        tf_model.compile(optimizer=Adam(config.LR),
                         loss=config.LOSS_FN,
                         metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10)

        r = tf_model.fit([padded_encoder_input_seq,
                          padded_decoder_input_seq],
                         one_hot_targets,
                         epochs=config.NUM_EPOCHS,
                         batch_size=config.BATCH_SIZE,
                         validation_split=config.VALIDATION_SPLIT,
                         callbacks=[early_stop])

        # plot some data
        plt.title('Training vs Validation loss')
        plt.plot(r.history['loss'], label='loss')
        plt.plot(r.history['val_loss'], label='val_loss')
        plt.legend()
        plt.grid()
        plt.savefig(config.LOSS_PLOT)
        plt.show()


        # accuracies
        plt.title('Training vs Validation accuracy')
        plt.plot(r.history['acc'], label='acc')
        plt.plot(r.history['val_acc'], label='val_acc')
        plt.legend()
        plt.grid()
        plt.savefig(config.ACCURACY_PLOT)
        plt.show()


        # creating the second model
        # the one that will be used during inference stage

        inf_encoder_model = Model(encoder_input_layer, final_encoder_states)

        inf_decoder_input_h = Input(shape=(config.LSTM_HIDDEN_VECTORS,))
        inf_decoder_input_c = Input(shape=(config.LSTM_HIDDEN_VECTORS,))
        inf_decoder_states_input = [inf_decoder_input_h, inf_decoder_input_c]

        # giving one word as input
        inf_decoder_word_input = Input(shape=(1,))
        inf_x = decoder_embedding_layer(inf_decoder_word_input)

        inf_decoder_o, inf_decoder_h, inf_decoder_c = decoder_lstm(inf_x,
                                                    initial_state=inf_decoder_states_input)

        inf_decoder_states = [inf_decoder_h, inf_decoder_c]

        inf_decoder_o = decoder_dense_layer(inf_decoder_o)

        inf_decoder_model = Model([inf_decoder_word_input] + inf_decoder_states_input,
                                  [inf_decoder_o] + inf_decoder_states)


        return tf_model, inf_encoder_model, inf_decoder_model

    def get_translation(self):

        return


