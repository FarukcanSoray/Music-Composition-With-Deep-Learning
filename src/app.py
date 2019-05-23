import midi
from preprocess import loadPieces
from model import getPieceBatch, fit_lstm_model
from utils import noteStateMatrixToMidi
import numpy as np
from keras.models import model_from_json, load_model
from keras.utils import print_summary
import pickle
from random import randint, uniform

def load_weights(fname):
    return load_model(fname + ".hdf5")


def generate_music_with_random_composition(fname, modelname):
    model = load_weights(modelname)
    test_x, test_y = getPieceBatch(pieces, 16*8, 16, 16)

    output = np.zeros(78)
    for i in range(randint(20,30)):
        tick = randint(40,150)
        thresh = uniform(0.55, 1.3)
        for j in range(randint(5,15)):
            out = model.predict(test_x)
            first = out[0][0]
            first = np.nan_to_num(first, 0)
            first[first>=thresh] = 1
            first[first<thresh] = 0 
            output = np.vstack((output, first))
            test_x = out

    noteStateMatrixToMidi(output[78:], fname, tick)
            

    

def generate_music(fname, modelname, thresh, tick, length):
    model = load_weights(modelname)
    #print(print_summary(model))
    test_x, test_y = getPieceBatch(pieces, 16*8, 16, 16)
    a = test_x
    output = np.zeros(78)
    for i in range(length):
        out = model.predict(a)
        first = out[0][0]
        first[first>=thresh] = 1
        first[first<thresh] = 0 
        output = np.vstack((output, first))
        a = out

    #noteStateMatrixToMidi(output, fname, tick)
    #test_x, test_y = getPieceBatch(pieces, 16*8, 16, 16)
    a = test_x
    output = np.zeros(78)
    for i in range(length):
        out = model.predict(a)
        first = out[0][0]
        first[first>=thresh] = 1
        first[first<thresh] = 0 
        output = np.vstack((output, first))
        a = out


    noteStateMatrixToMidi(output, fname, tick)

#pieces = loadPieces("/home/farukcan/MIDI_DATASET", batch_len=16*8)
#pickle_out = open("dict.pickle","wb")
#pickle.dump(pieces, pickle_out)
#pickle_out.close()

def training():#rnn 11 lstm 23 
    model = fit_lstm_model(getPieceBatch(pieces, 16*8, 16, 9600), 16, 500, 128, "model11")


"""    model = fit_lstm_model(getPieceBatch(pieces, 16*8, 16, 9600), 16, 500, 512, "model7")

    model_json = model.to_json()
    with open("model6.json", "w") as json_file:
        json_file.write(model_json) 

    #3lstmlayer
    model = fit_lstm_model(getPieceBatch(pieces, 16*8, 16, 9600), 16, 500, 50, "model6")#%23

    model_json = model.to_json()
    with open("model6.json", "w") as json_file:
        json_file.write(model_json) 
   2lstmlayer  model = fit_lstm_model(getPieceBatch(pieces, 16*8, 16, 9600), 16, 500, 40, "model5")

    model_json = model.to_json()
    with open("model5.json", "w") as json_file:
        json_file.write(model_json)
   model = fit_lstm_model(getPieceBatch(pieces, 16*8, 16, 4000), 8, 500, 20, "model1")

    model_json = model.to_json()
    with open("model1.json", "w") as json_file:
        json_file.write(model_json)


    model = fit_lstm_model(getPieceBatch(pieces, 16*8, 16, 8000), 8, 1000, 20, "model2")

    model_json = model.to_json()
    with open("model2.json", "w") as json_file:
        json_file.write(model_json)

    model = fit_lstm_model(getPieceBatch(pieces, 16*8, 16, 4000), 8, 500, 30, "model3")

    model_json = model.to_json()
    with open("model3.json", "w") as json_file:
        json_file.write(model_json)

    model = fit_lstm_model(getPieceBatch(pieces, 16*8, 16, 9200), 8, 1000, 20, "model4")

    model_json = model.to_json()
    with open("model4.json", "w") as json_file:
        json_file.write(model_json)
"""
    



pickle_off = open("dict.pickle","rb")
pieces = pickle.load(pickle_off)

#training()
#generate_music("music_model10_threshold_0_08_track_04", "model10", 0.08,150)
"""generate_music("music_model10_threshold_0_08_track_2", "model10", 0.08, 150)
generate_music("music_model10_threshold_0_06_track_2", "model10", 0.06, 200)
generate_music("music_model10_threshold_0_1_track_2", "model10", 0.1, 90)
generate_music("music_model10_threshold_0_115_track_2", "model10", 0.115, 80)"""

generate_music("abcdef", "model10", 0.12, 75, 78)

#generate_music_with_random_composition("random_composed_model10_track01", "model10")