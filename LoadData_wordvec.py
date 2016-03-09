import numpy as np
import theano

def load_embedding_to_dict(file_name):
    print "...loading file to dictionary"
    datafile = open(file_name, mode="r")
    dict = {}
    for line in datafile:
        line = line[:-1]
        line = line.strip()
        words = line.split(" ")
        vec = [float(word) for word in words[1:]]
        dict[words[0]] = vec

    print "Total words number: %i" % len(dict)

    return dict

def load_data(embedding_file='data/RNN_dict.txt',
              input_file=''):
    dict = load_embedding_to_dict(embedding_file)
    print "...loading RNN input file"
    datafile= open(input_file, mode='r')
    x_list = []
    y_list = []

    for line in datafile:
        line = line[:-1]
        line = line.strip()
        # sentence_matrix = [dict[word] for word in line.split(" ")]

        sentence_matrix = []
        sentence_matrix.append([0.] * 100)
        for word in line.split(" "):
            if dict.has_key(word):
                sentence_matrix.append(dict[word])
            else:
                sentence_matrix.append([0.] * 100)

        sentence_matrix.append([0.] * 100)

        matrix = np.asarray(sentence_matrix, dtype=theano.config.floatX)

        x_list.append(matrix[:-1])
        y_list.append(matrix[1:])

    x_list = np.asarray(x_list)
    y_list = np.asarray(y_list)

    return x_list, y_list

if __name__ == '__main__':
    x_set, y_set = load_data(input_file='data/RNNinput.txt')
    print x_set.shape, y_set.shape
