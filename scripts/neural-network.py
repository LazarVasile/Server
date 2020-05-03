import os
import json
import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from input_parser import input_parser

dataDir = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__),'../data')))

books = []

for filename in os.listdir(dataDir):
    if filename.endswith('.json'):
        books.append(filename)

def train_neural_networks():
    for filename in books:

        if os.path.exists(os.path.join(dataDir,filename.replace(".json", '_') + "model.yaml")):
            continue

        with open(os.path.join(dataDir,filename),"r", encoding="utf-8") as f:
            book_info = json.load(f)
        
        keys = [key for key in book_info.keys() if "l" not in key]

        training_data = []

        for fragment_id in keys:
            training_data.append({"id" : fragment_id, "fragment" : book_info[fragment_id]})

        #print ("%s fragments in training data" % len(training_data))

        words = []
        ids = []
        documents = []
        ignore_words = ['?', ',', '.', '!', ':', ';', '-']
        # loop through each fragment in our training data
        for pattern in training_data:
            # parse each word in the fragment
            #w = input_parser(pattern['fragment'])
            w = book_info[pattern['id'] + "_l"]
            # add to our words list
            words.extend(w)
            # add to documents in our corpus
            documents.append((w, pattern['id']))
            # add to our ids list
            if pattern['id'] not in ids:
                ids.append(pattern['id'])

        # stem and lower each word and remove duplicates
        words = [w for w in words if w not in ignore_words]
        words = sorted(list(set(words)))

        # remove duplicates
        #ids = list(set(ids))

        #print (len(documents), "documents")
        #print (len(ids), "ids", ids)
        #print (len(words), "unique stemmed words", words)


        # create our training data
        training = []
        output = []
        # create an empty array for our output
        output_empty = [0] * len(ids)

        # training set, bag of words for each sentence
        for doc in documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # create our bag of words array
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            training.append(bag)
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[ids.index(doc[1])] = 1
            output.append(output_row)

        #print ("# words", len(words))
        #print ("# ids", len(ids))

        x = np.array(training)
        y = np.array(output)
        hidden_neurons = int(math.sqrt(len(x[0]))) * 2

        model = Sequential()

        model.add(Dense(hidden_neurons, input_dim=len(x[0]), activation='relu'))
        model.add(Dense(hidden_neurons, activation='relu'))
        model.add(Dense(len(x), activation='sigmoid'))


        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(x, y, epochs=700)

        scores = model.evaluate(x, y, verbose=0)
        #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        # serialize model to YAML
        model_yaml = model.to_yaml()

        with open(os.path.join(dataDir,filename.replace(".json",'_') + "model.yaml"), "w") as yaml_file:
            yaml_file.write(model_yaml)

        # serialize weights to HDF5
        model.save_weights(os.path.join(dataDir,filename.replace(".json",'_') + "model.h5"))

        with open(os.path.join(dataDir,filename.replace(".json",'_') + "words.txt"),"w",encoding="utf-8") as f:
            f.write(json.dumps(words))

        print("Saved model to disk")

        del book_info
        del training_data
        del words
        del ids
        del documents
        del training
        del output
        del x
        del y
        del model
        del scores
        del model_yaml
        
        break

import time

nr_carti = 0

while nr_carti < 100:
    train_neural_networks()
    time.sleep(4)
    nr_carti += 1
