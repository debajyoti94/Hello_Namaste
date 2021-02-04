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
        fr_obj = feature_engg.DataPreprocessing()
        input_seq_encoder, input_seq_decoder,\
        output_seq_decoder, input_tokenizer,\
        translation_tokenizer = fr_obj.cleaning_data(config.RAW_DATASET)

        # getting fixed length padded sequences
        padded_encoder_input_seq, \
        padded_decoder_input_seq, \
        padded_decoder_output_seq,\
        max_encoder_input_seq_len, \
        max_decoder_input_seq_len= fr_obj.get_fixed_length_sequences(input_seq_encoder, input_seq_decoder, output_seq_decoder)

        dl_obj.dump_file(config.INPUT_SEQ_ENCODER, input_seq_encoder)
        dl_obj.dump_file(config.INPUT_SEQ_DECODER, input_seq_decoder)
        dl_obj.dump_file(config.OUTPUT_SEQ_DECODER, output_seq_decoder)
        dl_obj.dump_file(config.INPUT_TOKENIZER, input_tokenizer)
        dl_obj.dump_file(config.TRANSLATION_TOKENIZER, translation_tokenizer)

        dl_obj.dump_file(config.PADDED_ENCODER_INPUT_SEQ, padded_encoder_input_seq)
        dl_obj.dump_file(config.PADDED_DECODER_INPUT_SEQ, padded_decoder_input_seq)
        dl_obj.dump_file(config.PADDED_DECODER_OUTPUT_SEQ, padded_decoder_output_seq)
        dl_obj.dump_file(config.MAX_ENCODER_INPUT_SEQ_LEN, max_encoder_input_seq_len)
        dl_obj.dump_file(config.MAX_DECODER_INPUT_SEQ_LEN, max_decoder_input_seq_len)

    elif args.train == 'nmt':

        # train the model here



        # train the model here
        pass

    elif args.test == 'translation':

        # get hindi translation here
        pass