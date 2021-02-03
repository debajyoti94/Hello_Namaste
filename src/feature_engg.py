''' Here we apply data preprocessing techniques on the raw dataset '''

# import modules here
import config

import abc

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle

from matplotlib import pyplot as plt
import numpy as np

# create abc class here
class MustHaveForDP:

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def cleaning_data(self):
        '''
        In this function we take the input data and transform it to a
         format that can be used for the model to train
        :return:
        '''
        return


# create data preprocessing class here
class DataPreprocessing(MustHaveForDP):

    def cleaning_data(self, dataset_path):
        '''

        :param dataset_path:
        :return:
        '''

        # loading raw data here
        sample_id = 0
        input_text_encoder_list = []
        input_text_decoder_list = []
        output_text_decoder_list = []
        for line in open(dataset_path):

            sample_id += 1
            if sample_id > config.MAX_SAMPLES:
                break

            if config.SEQUENCE_DELIMITER not in line:
                sample_id -= 1
                # deducting the sample id that has been added above
                continue

            input_line, translated_line, *not_nedded = line.split(config.SEQUENCE_DELIMITER)

            # since we are implementing the seq2seq architecture
            # we will need 2 models, 1st model will be trained by teacher forcing methode
            # in the second model we will use the layers trained from the first model
            input_text = '<sos> ' + input_line
            input_text_decoder = '<sos> ' + translated_line
            output_text_decoder = translated_line + ' <eos>'

            input_text_encoder_list.append(input_text)
            input_text_decoder_list.append(input_text_decoder)
            output_text_decoder_list.append(output_text_decoder)

        # tokenize the input data
        input_tokenizer = Tokenizer(num_words=config.MAX_VOCAB_SIZE, filters='')
        input_tokenizer.fit_on_texts(input_text_encoder_list)
        input_seq_encoder = input_tokenizer.texts_to_sequences(input_text_encoder_list)

        # tokenize the translated data
        translation_tokenizer = Tokenizer(num_words=config.MAX_VOCAB_SIZE, filters='')
        translation_tokenizer.fit_on_texts(input_text_decoder_list + output_text_decoder_list)
        input_seq_decoder = translation_tokenizer.texts_to_sequences(input_text_decoder_list)

        return input_seq_encoder, input_seq_decoder, input_tokenizer, translation_tokenizer


    # this function will help in creating fixed length sequences
    # we will apply padding
    def get_fixed_length_sequences(self, encoder_input_seq, decoder_input_seq, decoder_output_seq):
        '''

        :param encoder_input_seq:
        :param decoder_input_seq:
        :param decoder_output_seq:
        :return:
        '''

        max_encoder_input_seq_len = max(len(s) for s in encoder_input_seq)
        padded_encoder_input_seq = pad_sequences(encoder_input_seq, maxlen=max_encoder_input_seq_len,
                                                 padding='pre')

        max_decoder_input_seq_len = max(len(s) for s in decoder_input_seq)
        padded_decoder_input_seq = pad_sequences(decoder_input_seq, maxlen=max_decoder_input_seq_len,
                                                 padding='post')

        padded_decoder_output_seq = pad_sequences(decoder_output_seq, maxlen=max_decoder_input_seq_len,
                                                  padding='post')

        return padded_encoder_input_seq, padded_decoder_input_seq, padded_decoder_output_seq


    def create_embedding_matrix(self, enocder_input_tokenizer):
        '''

        :param enocder_input_tokenizer:
        :return:
        '''
        # creatng embedding matrix
        word_vector_dict = {}
        word2idx = enocder_input_tokenizer.word_index
        embedding_matrix = np.zeros(shape=(len(word2idx), config.EMBEDDING_DIM), dtype='float32')

        # first we need to load the pretrained embedding file and create a word_vector_dict
        with open(config.PRETRAINED_EMBEDDINGS) as file_handle:
            for line in file_handle.split():
                try:
                    word = line[0]
                    vector = np.asarray(line[1:], dtype='float32')
                    word_vector_dict[word] = vector
                except ValueError:
                    continue

        # now that we have the word vector dict
        # we can create an embedding matrix for the words that exists in our dataset
        for word, index in word2idx:
            word_vector = word_vector_dict.get(word)
            if word_vector is not None:
                embedding_matrix[index] = word_vector


        return embedding_matrix, word_vector_dict


class DumpLoadFile:

    def dump_file(self, filename, *file):
        '''

        :param filename:
        :param file:
        :return:
        '''
        with open(filename, 'wb') as pickle_handle:
            pickle.dump(file, pickle_handle)


    def load_file(self, filename):
        '''

        :param filename:
        :return:
        '''
        with open(filename, 'rb') as pickle_handle:
            return pickle.load(pickle_handle)

