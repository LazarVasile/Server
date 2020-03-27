import nltk
from nltk.stem import SnowballStemmer

import os
import json
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

stemmer = SnowballStemmer("romanian")

dataDir = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__),'../data')))

books = []

for filename in os.listdir(dataDir):
    if filename.endswith('.json'):
        books.append(filename)

for filename in books:

    with open(os.path.join(dataDir,filename + "training.txt"),"r") as f:
        training = json.loads(f.read())
    with open(os.path.join(dataDir,filename + "output.txt"),"r") as f:
        output = json.loads(f.read())
    with open(os.path.join(dataDir,filename + "words.txt"),"r") as f:
        words = json.loads(f.read())
    
    x = np.array(training)
    y = np.array(output)
    hidden_neurons = 10

    model = Sequential()

    model.add(Dense(hidden_neurons, input_dim=len(x[0]), activation='relu'))
    #model.add(Dense(len(x[0]), activation='relu'))
    model.add(Dense(len(x), activation='sigmoid'))


    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x, y, epochs=25)

    scores = model.evaluate(x, y, verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(os.path.join(dataDir,filename + "model.yaml"), "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(os.path.join(dataDir,filename + "model.h5"))
    print("Saved model to disk")
