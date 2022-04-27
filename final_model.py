import numpy as np
import pandas as pd
import datetime, os
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split

# tensorflow imports
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorboard.plugins.hparams import api as hp

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Embedding
from tensorflow.keras.losses import sparse_categorical_crossentropy
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import GRU, Input, Dense, TimeDistributed, LSTM

#Tokenization
def tokenize(x):
  """
  Tokenize x
  :param x: List of sentences/strings to be tokenized
  :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
  """
  # TODO: Implement
  x_tk = Tokenizer()
  x_tk.fit_on_texts(x)

  return x_tk.texts_to_sequences(x), x_tk

#Padding
def pad(x, length=None):
  """
  Pad x
  :param x: List of sequences.
  :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
  :return: Padded numpy array of sequences
  """
  # TODO: Implement
  if length is None:
    length = max([len(sentence) for sentence in x])
  return pad_sequences(x, maxlen=length, padding='post', truncating='post')

def preprocess(x, y):
  """
  Preprocess x and y
  :param x: Feature List of sentences
  :param y: Label List of sentences
  :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
  """
  preprocess_x, x_tk = tokenize(x)
  preprocess_y, y_tk = tokenize(y)

  preprocess_x = pad(preprocess_x)
  preprocess_y = pad(preprocess_y)

  # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
  preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

  return preprocess_x, preprocess_y, x_tk, y_tk

#Loading the training dataset
df = pd.read_json('https://raw.githubusercontent.com/kkityeungg/CE888_datascience_finalproject/main/spider/train_spider.json')
x = df['question']
y = df['query']

#Preprocessing the dataset and output their processed data and tokenier
preprocess_x, preprocess_y, x_tk, y_tk = preprocess(x, y)

x_sequence_length = preprocess_x.shape[1]
y_sequence_length = preprocess_y.shape[1]
x_vocab_size = len(x_tk.word_index)
y_vocab_size = len(y_tk.word_index)

#Split the dateset to training and testing set
x_train, x_test, y_train, y_test = train_test_split(preprocess_x, preprocess_y, test_size=0.3, random_state=99)

def model_final(input_shape, output_sequence_length, question_vocab_size, query_vocab_size):
  """
  Build and train a embedding and bidirectional LSTM on x and y
  :param input_shape: Tuple of input shape
  :param output_sequence_length: Length of output sequence
  :param question_vocab_size: Number of unique words in the question dataset
  :param query_vocab_size: Number of unique words in the query dataset
  :return: Keras model built, but not trained
  """
  inputs = Input(shape=input_shape[1:])
  emb = Embedding(question_vocab_size, 100)(inputs)
  en_lstm_layer1 = Bidirectional(LSTM(128, return_sequences=True))(emb)
  en_lstm_layer2 = Bidirectional(LSTM(128, return_sequences=False))(en_lstm_layer1)
  final_enc = Dense(256, activation='relu')(en_lstm_layer2)
    
  dec1 = RepeatVector(output_sequence_length)(final_enc)
  de_lstm_layer1 = Bidirectional(LSTM(256, dropout=0.5, return_sequences=True))(dec1)
  layer = TimeDistributed(Dense(query_vocab_size, activation='softmax'))
  final = layer(de_lstm_layer1)

  model = Model(inputs=inputs, outputs=final)
  model.compile(loss=sparse_categorical_crossentropy, 
                optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
                metrics=['accuracy'])
  return model

model_final = model_final(
    x_train.shape,
    y_train.shape[1],
    x_vocab_size+1,
    y_vocab_size+1)

callbacks = [EarlyStopping(monitor='val_loss', patience=1)]

model_final.summary()
model_final_history = model_final.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test), callbacks = callbacks)

print(model_final_history.history.keys())
# summarize history for accuracy
plt.plot(model_final_history.history['accuracy'])
plt.plot(model_final_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('final_accuracy.png')
plt.show()

# summarize history for loss
plt.plot(model_final_history.history['loss'])
plt.plot(model_final_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('final_loss.png')
plt.show()

#Output the result of prediction with an example
y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
y_id_to_word[0] = '<PAD>'

sentence = 'how many heads of the departments are older than 56'
sentence = [x_tk.word_index[word] for word in sentence.split()]
sentence = pad_sequences([sentence], maxlen=x_train.shape[-1], padding='post')
sentences = np.array([sentence[0], x_train[0]])
predictions = model_final.predict(sentences, len(sentences))
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))