import os
import json
import fnmatch
from input_parser import input_parser
from textwrap import wrap

booksDir = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__),'../../DB')))
dataDir = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__),'../data')))

books = []

for filename in os.listdir(booksDir):
    if filename.endswith('.txt') and fnmatch.fnmatch(filename,"2???_a_*"):
        books.append(filename)
    if len(books) > 100:
        break

for filename in books:
    book_content = open(os.path.join(booksDir,filename),encoding="iso8859_2").read()

    if os.path.exists(os.path.join(dataDir,filename.replace('.txt','') + ".json")):
        os.remove(os.path.join(dataDir,filename.replace('.txt','') + ".json"))
    
    with open(os.path.join(dataDir,filename.replace('.txt','') + ".json"), 'w', encoding="iso8859_2") as fd:
        my_dict = {}
        i = 1

        propozitii = book_content.split('.')
        k = 0

        while k < len(propozitii):
            try:
                cuvinte_fragment = 0 #cuvinte in fragmentul curent
                fragment = ""
                
                while cuvinte_fragment < 150 and k < len(propozitii):
                    cuvinte = propozitii[k].split(' ') #cuvinte in propozitia k
                    cuvinte_fragment += len(cuvinte) #adaugam propozitia in fragment
                    fragment += propozitii[k]
                    k += 1    #adaugam propozitii in fragment pana ajungem la 150 de cuvinte
                
                my_dict[i] = fragment
                i += 1
            except:
                pass

        json.dump(my_dict, fd, indent=4, ensure_ascii=False)
    
""" 
books = []

for filename in os.listdir(dataDir):
    if filename.endswith('.json'):
        books.append(filename)

book_info = {}

for filename in books:
    with open(os.path.join(dataDir,filename), 'r', encoding="iso8859_2") as fd:
        book_info = json.load(fd)
    

    training_data = []

    for fragment_id in book_info.keys():
        training_data.append({"id" : fragment_id, "fragment" : book_info[fragment_id]})

    #print ("%s fragments in training data" % len(training_data))

    words = []
    ids = []
    documents = []
    ignore_words = ['?', ',', '.', '!', ':', ';', '-']
    # loop through each fragment in our training data
    for pattern in training_data:
        # tokenize each word in the fragment
        w = input_parser(pattern['fragment'])
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

    import json

    with open(os.path.join(dataDir,filename + "training.txt"),"w",encoding="iso8859_2") as f:
        f.write(json.dumps(training))
    with open(os.path.join(dataDir,filename + "output.txt"),"w",encoding="iso8859_2") as f:
        f.write(json.dumps(output))
    with open(os.path.join(dataDir,filename+ "words.txt"),"w",encoding="iso8859_2") as f:
        f.write(json.dumps(words)) 
 """
