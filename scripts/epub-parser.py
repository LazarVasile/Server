import os
import json

from epub_conversion.utils import open_book, convert_epub_to_lines

booksDir = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__),'../books')))
dataDir = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__),'../data')))


books = []

for filename in os.listdir(booksDir):
    if filename.endswith('.epub'):
        books.append(filename)


import ebooklib
from ebooklib import epub

import re
def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

for filename in books:
    book = epub.read_epub(os.path.join(booksDir,filename))

    if os.path.exists(os.path.join(dataDir,filename.replace('.epub','') + ".json")):
        os.remove(os.path.join(dataDir,filename.replace('.epub','') + ".json"))

    with open(os.path.join(dataDir,filename.replace('.epub','') + ".json"), 'w', encoding="iso8859_2") as fd:
        
        my_dict = {}
        i = 1

        lista = [doc for doc in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]

        for j in range(1,len(lista)):
            page = lista[j].content
        
            body_content = re.compile(r'<body.*?>(.*?)</body>', re.DOTALL).findall(page.decode("iso8859_2"))

        

            paragraph_list = re.compile(r'<p.*?>(.*?)</p>', re.DOTALL).findall(body_content[0])

            for k in range(len(paragraph_list)):
                my_dict[i] = striphtml(paragraph_list[k])
                i = i + 1
        
        json.dump(my_dict, fd, indent=4, ensure_ascii=False)


import nltk
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("romanian")

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
        w = nltk.word_tokenize(pattern['fragment'])
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, pattern['id']))
        # add to our ids list
        if pattern['id'] not in ids:
            ids.append(pattern['id'])

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = list(set(words))

    # remove duplicates
    ids = list(set(ids))

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
        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
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

    with open(os.path.join(dataDir,filename + "training.txt"),"w") as f:
        f.write(json.dumps(training))
    with open(os.path.join(dataDir,filename + "output.txt"),"w") as f:
        f.write(json.dumps(output))
    with open(os.path.join(dataDir,filename+ "words.txt"),"w") as f:
        f.write(json.dumps(words)) 

