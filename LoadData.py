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

def load_data_x(x_file='data/test_1/RNNwords.txt',
                embedding_file='data/RNN_dict.txt'):
    x, x_ = load_data(input_file=x_file, embedding_file=embedding_file)

    return x

def load_data_y(y_file='data/RNNlabels.txt'):
    ret_y = []
    for line in open(y_file):
        line = line.strip()
        ret_y.append(int(line))

    return ret_y

def load_data_xy(x_file='data/RNNinput.txt',
              y_file='data/MLPLabels.txt',
              embedding_file='data/RNN_dict.txt'):
    x, x_ = load_data(input_file=x_file, embedding_file=embedding_file)
    Y_file = open(y_file, mode='r')
    y_ = []

    line_num = 0
    for line in Y_file:
        line = line.strip()
        y_.append(int(line))

    y = np.asarray(y_, dtype='int8')

    datasets = []
    datasets.append([x[4000:], y[4000:]])
    datasets.append([x[2000:4000], y[2000:4000]])
    datasets.append([x[:2000], y[:2000]])

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(value=data_x)
        shared_y = theano.shared(value=data_y)
        return shared_x, shared_y

    rval = []
    rval.append(shared_dataset(datasets[0]))
    rval.append(shared_dataset(datasets[1]))
    rval.append(shared_dataset(datasets[2]))

    return datasets


if __name__ == '__main__':
    x_set, y_set = load_data(input_file='data/RNNinput.txt')
    print x_set.shape, y_set.shape
