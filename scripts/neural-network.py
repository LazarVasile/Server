import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import json
from nltk.tokenize import word_tokenize
import re
import collections
import numpy as np
import random
import math
import datetime as dt
import tensorflow as tf
#tf.disable_v2_behavior()

dataDir = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__),'../data')))

punctuation = re.compile(r'[-.?!:;\'\&\"\`\\\/\|()]')

def clear_punctuation(lista_tokens):
    output_list = []

    for item in lista_tokens:
        word = punctuation.sub("", item)
        if len(word) > 0:
            output_list.append(word)

    return output_list

def lower_case_tokens(lista_tokens):
    output_list = []

    for item in lista_tokens:
        item = str.lower(item)
        output_list.append(item)
    
    return output_list

def build_dataset(words, n_words):
    count = [['UNK', -1]] #UNK - unknown
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionar = dict()
    for word, _ in count:
        dictionar[word] = len(dictionar)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionar:
            index = dictionar[word]
        else:
            index = 0 # dictionar[UNK]
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionar = dict(zip(dictionar.values(), dictionar.keys()))
    return data, count, dictionar, reversed_dictionar

def generate_batch(data, batch_size, num_skips, skip_window, data_index):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size,1), dtype=np.int32)
    span = 2 * skip_window + 1 # skip_window input_word skip_window
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window] # this is the input word
            context[i * num_skips + j, 0] = buffer[target] # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context




books = []

for filename in os.listdir(dataDir):
    if filename.endswith('.json'):
        books.append(filename)

for filename in books:
    file_content_dict = {}

    with open(os.path.join(dataDir,filename), 'r', encoding="iso8859_2") as fd:
        file_content_dict = json.load(fd)
    
    vocabulary = []

    for i in file_content_dict.keys():
        tokens = word_tokenize(file_content_dict[i])

        tokens = lower_case_tokens(tokens)

        tokens = clear_punctuation(tokens)

        vocabulary += tokens

    
    #print(vocabulary[:6])
    vocabulary_size = 10000

    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
    del vocabulary

    valid_size = 16
    valid_window = 100
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 10       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.


    # Input data.
    train_inputs = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
    train_context = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Look up embeddings for inputs.
    embeddings = tf.Variable(tf.random.uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the softmax
    weights = tf.Variable(tf.random.truncated_normal([embedding_size, vocabulary_size],stddev=1.0 / math.sqrt(embedding_size)))
    biases = tf.Variable(tf.zeros([vocabulary_size]))
    hidden_out = tf.transpose(tf.matmul(tf.transpose(weights), tf.transpose(embed))) + biases

    # convert train_context to a one-hot format
    train_one_hot = tf.one_hot(train_context, vocabulary_size)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hidden_out, labels=train_one_hot))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.compat.v1.global_variables_initializer()


    num_steps = 1000
    data_index = 0

    with tf.compat.v1.Session() as session:
        # We must initialize all variables before we use them.
        session.run(init)
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_context = generate_batch(data,batch_size, num_skips, skip_window, data_index)
            feed_dict = {train_inputs: batch_inputs, train_context: batch_context}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, cross_entropy], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        try:
                            close_word = reverse_dictionary[nearest[k]]
                        except:
                            pass
                        log_str = '%s %s,' % (log_str, close_word)
                    #print(log_str)
        final_embeddings = normalized_embeddings.eval()

    if os.path.exists(os.path.join(dataDir,filename.replace('.json','') + ".txt")):
        os.remove(os.path.join(dataDir,filename.replace('.json','') + ".txt"))
        
    with open(os.path.join(dataDir,filename.replace('.json','') + ".txt"), "w") as fd:
        fd.write('# Array shape: {0}\n'.format(final_embeddings.shape))
        np.savetxt(fd, final_embeddings, fmt='%-7.2f')


    #To load the array from the file
    #final_embeddings = np.loadtxt(os.path.join(dataDir,filename.replace('.json','') + ".txt"))