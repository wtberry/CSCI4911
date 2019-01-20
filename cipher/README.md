# CSCI 4911, Toy Problem Vigenere cipher with LSTM

Bunch of experimental RNN / LSTMs, to try with vigenere ciphers. Some of them are practices for PyTorch, and others are different LSTM/RNN architectures.

## Experimental Models
* BLSTM.py: bidirectional LSTM
* LSTMCell.py: Single LSTM model
* mul/single_layer_mul/single_batch_lstm: LSTMs with multiple/single, processing multiple/single batch of encrypted sequences.

## PyTorch Practices
* torch_rnn.py: pytorch [RNN tutorial](https://www.youtube.com/watch?v=ogZi5oIo4fI&index=12&list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m 
) from youtube. This does not incorporate custom RNN class.
* torch_lstm_cipher.py: lstm model with custom LSTM class. Applied on a string of my name.
* torch_rnn_seq.py: rnn with custom torch rnn class.
* torch_rnn_seq_name.py: rnn model, using custom rnn class. 

## Miscellaneous
* pre_process.py: read in csv into dataframe, encode, and one hot the data.
* new_logs/: tensorboard logs for experimentational runs
* [logger.py](https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
): tensorboard logging script when using non-tf ML libraries

## vigenere/
* data/: data file, csv
* txt/: plain text and word list for keys
* data_create.py: create sequential data based on the plain text. Output is plain text and encrypted text csv