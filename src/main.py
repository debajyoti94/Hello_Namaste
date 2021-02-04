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
        print('Creating input/output sequences using tokenizer ...')
        input_seq_encoder, \
        input_seq_decoder,\
        output_seq_decoder,\
        input_tokenizer,\
        translation_tokenizer,\
        actual_vocab_size_input_encoder,\
        actual_vocab_size_input_decoder= fr_obj.cleaning_data(config.RAW_DATASET)

        # getting fixed length padded sequences
        print('Padding/making fixed length sequences ...')
        padded_encoder_input_seq, \
        padded_decoder_input_seq, \
        padded_decoder_output_seq,\
        max_encoder_input_seq_len, \
        max_decoder_input_seq_len= fr_obj.get_fixed_length_sequences(input_seq_encoder, input_seq_decoder, output_seq_decoder)

        # getting the embedding matrix
        print('Creating the embedding matrix ...')
        embedding_matrix, word_vector_dict = fr_obj.create_embedding_matrix(input_tokenizer)

        dl_obj.dump_file(config.INPUT_SEQ_ENCODER, input_seq_encoder)
        dl_obj.dump_file(config.INPUT_SEQ_DECODER, input_seq_decoder)
        dl_obj.dump_file(config.OUTPUT_SEQ_DECODER, output_seq_decoder)
        dl_obj.dump_file(config.INPUT_TOKENIZER, input_tokenizer)
        dl_obj.dump_file(config.TRANSLATION_TOKENIZER, translation_tokenizer)
        dl_obj.dump_file(config.ACTUAL_VOCAB_SIZE_INPUT, actual_vocab_size_input_encoder)
        dl_obj.dump_file(config.ACTUAL_VOCAB_SIZE_TRANSLATION, actual_vocab_size_input_decoder)

        dl_obj.dump_file(config.PADDED_ENCODER_INPUT_SEQ, padded_encoder_input_seq)
        dl_obj.dump_file(config.PADDED_DECODER_INPUT_SEQ, padded_decoder_input_seq)
        dl_obj.dump_file(config.PADDED_DECODER_OUTPUT_SEQ, padded_decoder_output_seq)
        dl_obj.dump_file(config.MAX_ENCODER_INPUT_SEQ_LEN, max_encoder_input_seq_len)
        dl_obj.dump_file(config.MAX_DECODER_INPUT_SEQ_LEN, max_decoder_input_seq_len)

        dl_obj.dump_file(config.EMBEDDING_MATRIX, embedding_matrix)
        dl_obj.dump_file(config.WORD_VECTOR_DICT, word_vector_dict)

    elif args.train == 'nmt':

        # train the model here
        padded_encoder_input_seq = dl_obj.load_file(config.PADDED_ENCODER_INPUT_SEQ)
        padded_decoder_input_seq = dl_obj.load_file(config.PADDED_DECODER_INPUT_SEQ)
        padded_decoder_output_seq = dl_obj.load_file(config.PADDED_DECODER_OUTPUT_SEQ)
        embedding_matrix = dl_obj.load_file(config.EMBEDDING_MATRIX)
        actual_vocab_size_input_encoder = dl_obj.load_file(config.ACTUAL_VOCAB_SIZE_INPUT)
        actual_vocab_size_input_decoder = dl_obj.load_file(config.ACTUAL_VOCAB_SIZE_TRANSLATION)
        max_input_seq_encoder_len = dl_obj.load_file(config.MAX_ENCODER_INPUT_SEQ_LEN)
        max_input_seq_decoder_len = dl_obj.load_file(config.MAX_DECODER_INPUT_SEQ_LEN)

        tm_obj = model.TranslationModel()

        # train the model here
        tf_model, inf_encoder_model, inf_decoder_model = tm_obj.train_translation_model(
            padded_encoder_input_seq[0],
            padded_decoder_input_seq[0],
            padded_decoder_output_seq[0],
            embedding_matrix[0],
            actual_vocab_size_input_encoder[0],
            actual_vocab_size_input_decoder[0],
            max_input_seq_encoder_len[0],
            max_input_seq_decoder_len[0]
        )

        tf_model.save_weights(config.TEACHER_FORCING_MODEL)
        inf_encoder_model.save_weights(config.INFERENCE_ENCODER_MODEL)
        inf_decoder_model.save_weights(config.INFERENCE_DECODER_MODEL)

    elif args.test == 'translation':
        # in order to use the models that we have saved through training
        # we need to recreate these models and then load the weights

        # get hindi translation here
        pass