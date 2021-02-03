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
        # tokenize the text here

        # create input and output sequences

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


