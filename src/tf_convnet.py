import tensorflow as tf
import gensim
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
import itertools
import argparse

# handle args
parser = argparse.ArgumentParser()
parser.add_argument("-wv", "--word_vecs", required=True, type=str, help="relative path of word2vec file.")
args = vars(parser.parse_args())

# import train and test sets, set params
df = pd.read_csv("../data/train.csv")
df_test = pd.read_csv("../data/test.csv")

pad_sym = ''
batch_size = 63
filter_sizes = [3, 4, 5]
n_filters = 100
n_epochs = 50


# process data into form suitable for convnet
def tokenize_trunc_pad(dataframe, pad_sym, max_len=48):
    """Tokenize list, truncate if llonger than 48, pad to length 48 if shorter."""

    dataframe['line_tokens'] = dataframe.line.apply(lambda x: word_tokenize(x.replace("-", "").lower()))
    sent_tokens = [s[:max_len] if len(s) > max_len else s for s in dataframe.line_tokens.tolist()]
    dataframe['trunc_lines'] = sent_tokens
    dataframe['trunc_lines'] = dataframe['trunc_lines'].apply(lambda x: x + [pad_sym] * (max_len - len(x)))

    return dataframe


df = tokenize_trunc_pad(df, pad_sym=pad_sym)
df_test = tokenize_trunc_pad(df_test, pad_sym=pad_sym)


# get token embeddings and create dictionary for token to embedding index and character to class label
# line dicts
count_dict = Counter(itertools.chain.from_iterable(df['trunc_lines'].tolist()))
index_dict = dict([(w, i) for (i, w) in enumerate(count_dict.keys())])
index_dict[pad_sym] = len(index_dict)  # add padding symbol used above

# char dict
char_dict = dict([(char, i) for (i, char) in enumerate(df.character.unique())])
n_chars = len(char_dict)

# word2vec embeddings
word_vecs = gensim.models.KeyedVectors.load_word2vec_format(args['word_vecs'])
embedding_matrix = np.zeros(shape=(len(index_dict) + 1, 100))  # the matrix to store the token embeddings
for token in index_dict.keys():
    row_n = index_dict[token]
    try:
        embedding_matrix[row_n] = word_vecs.word_vec(token)
    except KeyError:
        embedding_matrix[row_n] = np.random.normal(scale=0.5, size=100)


# convert tokens to indices of embedding_matrix
# some tokens in training set won't be in dictionary
def token_embedding_index(token_list, token_index_dict):
    """Return list of indices of tokens in token_list for embedding_matrix. Use padding embedding if not in dict."""

    return [token_index_dict.get(t) if t in token_index_dict.keys() else token_index_dict.get('') for t in token_list]


df['int_lines'] = df['trunc_lines'].apply(lambda x: token_embedding_index(x, index_dict))
df['int_chars'] = df['character'].apply(char_dict.get)
df_test['int_lines'] = df_test['trunc_lines'].apply(lambda x: token_embedding_index(x, index_dict))
df_test['int_chars'] = df_test['character'].apply(char_dict.get)


# produce numpy array for training and test set labels
train_label_matrix = np.zeros(shape=(len(df.int_chars), len(char_dict)))
for i in range(len(df.int_chars)):
    train_label_matrix[i, df.int_chars[i]] = 1

test_label_matrix = np.zeros(shape=(len(df_test.int_chars), len(char_dict)))
for i in range(len(df_test.int_chars)):
    test_label_matrix[i, df_test.int_chars[i]] = 1


# create tf graph
def conv_maxpool_layer(filter_size, n_filters, conv_input):
    """Produce max pooling layer for fixed filter size with specified number of filters."""

    filter_weights = tf.Variable(initial_value=tf.truncated_normal(shape=[filter_size, 100, 1, n_filters]),
                                 name='filter_weights',
                                 dtype=tf.float32)
    bias = tf.Variable(initial_value=tf.zeros(shape=[n_filters]),
                       name='bias')
    conv = tf.nn.conv2d(input=conv_input,
                        filter=filter_weights,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv')
    conv_relu = tf.nn.relu(features=conv + bias,
                           name='relu')
    max_pool = tf.nn.max_pool(value=conv_relu,
                              ksize=[1, 49 - filter_size, 1, 1],  # recall conv output is of size height-filter_size+1
                              strides=[1, 1, 1, 1],  # applying max pooling line-wise
                              padding='VALID',
                              name='max_pool')

    return max_pool


graph = tf.Graph()

with graph.as_default():
    # define inputs
    lines = tf.placeholder(tf.int32, name='lines')
    lines_labels = tf.placeholder(tf.float32, name='lines_labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # to control drop out for training/prediction

    # define embeddings layer
    embeddings = tf.Variable(initial_value=embedding_matrix, trainable=True, name='embeddings')
    lines_embedding = tf.nn.embedding_lookup(embeddings, lines)
    lines_embedding = tf.expand_dims(lines_embedding, -1)  # we need to add a channel dim (dim=1)
    lines_embedding = tf.cast(lines_embedding, dtype=tf.float32)

    # define convolutional layers using helper function
    conv_outputs = []
    for i in filter_sizes:
        with tf.name_scope(
                'conv_filter_size_{}'.format(i)):  # have to make layers within scope of different convolutions
            conv_outputs.append(conv_maxpool_layer(filter_size=i, n_filters=n_filters, conv_input=lines_embedding))

    # concatenate the convolution outputs into a single vector of length 3*n_filters
    concat_output = tf.concat(values=conv_outputs, axis=3)
    concat_output = tf.reshape(tensor=concat_output, shape=[-1, 3 * n_filters])

    # make fully connected layer w/ dropout
    dropout = tf.nn.dropout(concat_output, keep_prob=keep_prob)
    fc_weights = tf.Variable(tf.truncated_normal(shape=[3 * n_filters, n_chars]),
                             name='fc_weights')
    fc_bias = tf.Variable(tf.zeros(shape=[n_chars]), name='fc_bias')
    logits = tf.matmul(dropout, fc_weights) + fc_bias

    # prediction and loss
    softmax_output = tf.nn.softmax(logits)
    predictions = tf.argmax(softmax_output, axis=1)
    accuracy = tf.metrics.accuracy(tf.argmax(lines_labels, axis=1), predictions, name='accuracy')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=lines_labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


# create and run session
X_train = np.array(df.int_lines.tolist())
y_train = train_label_matrix
X_test = np.array(df_test.int_lines.tolist())
y_test = test_label_matrix

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())  # initialize global variables
    sess.run(tf.local_variables_initializer())  # necessary for accuracy
    saver = tf.train.Saver()  # create object to save model after training

    for epoch in range(n_epochs):
        i_shuffle = np.random.permutation(X_train.shape[0])
        X_train = X_train[i_shuffle]
        y_train = y_train[i_shuffle]
        for batch in range(X_train.shape[0] // batch_size):
            X_batch = X_train[batch * batch_size:(batch + 1) * batch_size]
            y_batch = y_train[batch * batch_size:(batch + 1) * batch_size]

            feed_dict = {lines: X_batch, lines_labels: y_batch, keep_prob: 0.5}
            _, batch_loss = sess.run([optimizer, loss], feed_dict=feed_dict)

        _, train_accuracy = sess.run(accuracy, feed_dict={lines: X_train, lines_labels: y_train, keep_prob: 1})
        print("Training accuracy after epoch {}: {}".format(epoch, train_accuracy))

        if epoch % 10 == 0:  # get test accuracy every 10 epochs
            _, test_accuracy = sess.run(accuracy, feed_dict={lines: X_train, lines_labels: y_train, keep_prob: 1})
            print("Test accuracy after epoch {}: {}".format(epoch, test_accuracy))

    # view training accuracy of trained model
    _, test_accuracy = sess.run(accuracy, feed_dict={lines: X_train, lines_labels: y_train, keep_prob: 1})
    print("Final test accuracy of trained model is {}".format(test_accuracy))
    saver.save(sess, save_path="../results/convnet/convnet-50")

    # save softmax output for training and test set
    train_softmax_array = sess.run(softmax_output, feed_dict={lines:X_train, keep_prob: 1})
    test_softmax_array = sess.run(softmax_output, feed_dict={lines: X_train, keep_prob: 1})
    np.save("../results/convnet/train_softmax", train_softmax_array)
    np.save("../results/convnet/test_softmax", test_softmax_array)

