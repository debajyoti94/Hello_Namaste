''' Here we apply data preprocessing techniques on the raw dataset '''

# import modules here
import config

import abc

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle

from matplotlib import pyplot as plt

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

        max_encoder_input_seq_len = max(len(s) for s in encoder_input_seq)
        padded_encoder_input_seq = pad_sequences(encoder_input_seq, maxlen=max_encoder_input_seq_len,
                                                 padding='pre')

        max_decoder_input_seq_len = max(len(s) for s in decoder_input_seq)
        padded_decoder_input_seq = pad_sequences(decoder_input_seq, maxlen=max_decoder_input_seq_len,
                                                 padding='post')

        padded_decoder_output_seq = pad_sequences(decoder_output_seq, maxlen=max_decoder_input_seq_len,
                                                  padding='post')

        return padded_encoder_input_seq, padded_decoder_input_seq, padded_decoder_output_seq





        return


        return
    def create_embedding_matrix(self):
        # creatng embedding matrix
        return


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


dp_obj  = DataPreprocessing()
dp_obj.cleaning_data(config.RAW_DATASET)