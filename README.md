# Hello_Namaste:
### Neural Machine Translation model for translating English to Hindi Language.

This project is about building a Neural Machine Translation (NMT) model. The dataset used in this project is taken from [here](http://www.manythings.org/anki/).
Using the English-Hindi dataset.

### Architecture of the model: Sequence to Sequence (Seq2Seq) model with an encoder and decoder architecture.
In an encoder-decoder architecture, the encoder is responsible for taking the input and folding it up to a single thought vector. The decoder then takes this thought vector and produces an output in the desired format. We use LSTM (Long Short Term Memory) network for building this Sequence to Sequence model.

![lstm_whiteboard.jpg](https://github.com/debajyoti94/Hello_Namaste/blob/main/for_readme/lstm_whiteboard.jpg)

### Implementation details for building a Seq2Seq model:

* The Encoder: 
  - while buidling the encoder, we are interested in only the last hidden state of the LSTM unit. We are not considering the hidden states of the other LSTM units here.
  - Why so? Because we hope that the encoder will capture the relevant context from the previous hidden states into its vector.
  - Due to this reason, we do not consider the output states provided by the LSTM units in the encoder side.
  - Since the implementation is done in Keras, we keep ```return_sequences = False````.
  
* The Decoder:
  - The vector size of the LSTM must be the same as the encoder as the hidden state of the final LSTM goes as initial to the decoder LSTM.
  - Each output of the LSTM is passed through a dense layer with Softmax activation, using which we obtain a probability distribution of the output words.
  - To select the output of the decoder we take an argmax from the probability distribution.
  - While training the decoder, we apply Teacher Forcing method. In the teacher forcing method, we pass the expected output of the previous LSTM unit as the input to the current LSTM unit. We pass the true word as input and expect the same at the output.
  - During inference stage, the decoder will take 1 input at a time and predict one word at a time. We will pass the previous predicted output from the LSTM as the input to the decoder.

* As you can infer now, we need two different decoders, one during training and one during inference. This is why we create two different models. The first model is for training purpose and the second model will be used for sampling and using the decoded layers of the trained model. 

* While training the decoder, we need to pass a start of sentence <sos> token and an end of sentence token <eos> for the model to understand when to start and when to stop. During inference stage, we pass the decoder, the hidden states of the encoder and the <sos> token as input. The decoder stops predicting, when the token it has predicted is <eos>, indicating that it is the end of sentence.


### About the dataset:

As mentioned above, we use the English-Hindi translation dataset. The dataset consists of 2915 lines. Each line in the file consists of English sentence and its corresponding Hindi translation separated by a <TAB> space. For our experiment purpose we consider 2000 samples while training the model. 
  
### Repository structure:

The repository has multiple directories, with each serving a different purpose:
- input/: contains the processed data like:
        - Tokenizers used to fit the input and output data
        - Fixed length Sequences of input and output language used for training the model
        - Embedding matrix consisting of word vectors for each word in the input data
- model/: consists of the weights for the trained encoder and decoder model.
- plots/: consists of the train and accuracy plots, comparing train and validation set.
- src/: this directory consists of the source code for the project.
    - config.py: consists of variables which are used all across the code.
    - feature_engg.py: used for preprocessing the raw data and making the building blocks for training the model. All the components in the /input directory are produced via the functions in ```feature_engg.py```
    - test_functionalities: using pytest module, i define some data sanity checks on the training data.
    - model.py: this file contains the code for implementing the model. The train and the inference stage.
    - main.py: in this file all the code comes together and we define commandline arguments for performing each operation.
    
### To preprocess the raw data, use:
  ```python main.py --preprocess data```

### To train the model use the following command:
  ```python main.py --train nmt```
  
### To get translation for the sentence mentioned in main.py, use:
  ```python main.py --test translation```
    





