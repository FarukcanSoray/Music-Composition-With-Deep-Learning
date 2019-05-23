import random
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Softmax, Dropout, Bidirectional, Activation, SimpleRNN
from keras.layers.core import Flatten
from keras.backend import squeeze
from keras.callbacks import ModelCheckpoint

def getPieceSegment(pieces, batch_len, division_len):
    piece_output = random.choice(list(pieces.values()))
    start = random.randrange(0,len(piece_output)-batch_len,division_len)
    #print ("Range is {} {} {} -> {}".format(0,len(piece_output)-batch_len,division_len, start))

    seg_out = piece_output[start+1:start+batch_len]
    seg_in = piece_output[start:start+batch_len-1]#noteStateMatrixToInputForm(seg_out)

    return seg_in, seg_out

def getPieceBatch(pieces, batch_len, division_len, batch_width):
    i,o = zip(*[getPieceSegment(pieces, batch_len, division_len) for _ in range(batch_width)])
    return np.array(i), np.array(o)


def fit_lstm_model(train, batch_size, epoch, neurons, weight_filename):
    X, y = train
    model = Sequential()
    model.add(LSTM(neurons, recurrent_activation ='sigmoid',  batch_input_shape=(batch_size, X.shape[1], X.shape[2]), recurrent_dropout=0.2, stateful=True, return_sequences=True))
    model.add(LSTM(neurons, recurrent_dropout=0.2,  stateful=True, return_sequences=True))
    model.add(LSTM(neurons, recurrent_dropout=0.2,  stateful=True, return_sequences=True))
    model.add(LSTM(neurons, recurrent_dropout=0.2,  stateful=True, return_sequences=True))
    model.add(LSTM(neurons, recurrent_dropout=0.2,  stateful=True, return_sequences=True))
    
    #model.add(SimpleRNN(neurons,   batch_input_shape=(batch_size, X.shape[1], X.shape[2]), recurrent_dropout=0.2, stateful=True, return_sequences=True))
    #model.add(SimpleRNN(neurons, recurrent_dropout=0.2,  stateful=True, return_sequences=True))
    #model.add(LSTM(neurons,  stateful=True, return_sequences=True))
    #model.add(Dropout(0.5))
    #model.add(Dense(512))
    #model.add(Dropout(0.5))
    model.add(Dense(78))
    #model.add(Dropout(0.5))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics= ["accuracy"])
    filepath= weight_filename+".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    for i in range(epoch):
        model.fit(X, y, epochs=1, verbose =1, validation_split=0.15, batch_size=batch_size, callbacks=callbacks_list, shuffle=False)
        model.reset_states()
    return model