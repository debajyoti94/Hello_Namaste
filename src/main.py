''' This is where all the modules come together
 We train the model here and also call the inference stage module here'''


# import modules here
import config
import model
import feature_engg

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # define commandline arguments using argparse
    parser.add_argument('--preprocess', type=str,
                        help='Provide arguments \"--preprocess data\" '
                             'to clean the raw data and convert it to a usable format.')

    parser.add_argument('--train', type=str,
                        help='Provide arguments \"--train nmt\" '
                             'to train the Neural Machine Translation(NMT) model.')

    parser.add_argument('--test', type=str,
                        help='Provide arguments \"--test translation\" '
                             'to provide input in English and get translation in Hindi.')

    args = parser.parse_args()

    # creating object for loading and dumping pickled files
    dl_obj = feature_engg.DumpLoadFile()

    if args.preprocess == 'data':

        # call data preprocessing functions here
        pass

    elif args.train == 'nmt':


        # train the model here
        pass

    elif args.test == 'translation':

        # get hindi translation here
        pass