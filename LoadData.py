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
              embedding_file='data/RNN_dict.txt',
                 sample_sum=133914):
    x, x_ = load_data(input_file=x_file, embedding_file=embedding_file)
    Y_file = open(y_file, mode='r')
    y_ = []

    if not sample_sum is int:
        sample_sum = int(sample_sum)

    for line in Y_file:
        line = line.strip()
        y_.append(int(line))

    y = np.asarray(y_, dtype='int8')

    datasets = []
    datasets.append([x[:sample_sum-1], y[:sample_sum-1]])
    datasets.append([x[sample_sum-1:], y[sample_sum-1:]])

    print "Number of training examples: ", len(datasets[0][0])
    print "Number of test examples: ", len(datasets[1][1])

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(value=data_x)
        shared_y = theano.shared(value=data_y)
        return shared_x, shared_y

    rval = []
    rval.append(shared_dataset(datasets[0]))
    rval.append(shared_dataset(datasets[1]))

    return datasets

def load_data_right_wrong(embedding_file='../data/train/dict.txt',
                          input_file='../data/train/words.txt'):
    dict = load_embedding_to_dict(embedding_file)
    f = open(input_file, mode='r')

    right_list = []
    wrong_list = []

    num = 0
    for line in f:
        line = line.strip()
        sentence_matrix = []
        sentence_matrix.append([0.] * 100)
        for word in line.split(" "):
            if dict.has_key(word):
                sentence_matrix.append(dict[word])
            else:
                sentence_matrix.append([0.] * 100)
        matrix = np.asarray(sentence_matrix, dtype=theano.config.floatX)
        if num % 4 == 0:
            right_list.append(matrix)
        else:
            wrong_list.append(matrix)
        num += 1

    right_list = np.asarray(right_list)
    wrong_list = np.asarray(wrong_list)

    dataset = []
    dataset.append([right_list[:5931], wrong_list[:17793]])
    dataset.append([right_list[5931:], wrong_list[17793:]])

    return dataset

def load_data_qas(embedding_file='data/train/dict.txt',
                  word_file='data/train/words.txt',
                  label_file='data/train/labels.txt',
                  gap=7):
    dict = load_embedding_to_dict(embedding_file)
    f = open(word_file, mode='r')

    q_list = []
    a_list = []
    num = 0
    for line in f:
        line = line.strip()
        sentence_matrix = []
        sentence_matrix.append([0.] * 100)
        for word in line.split(" "):
            if dict.has_key(word):
                sentence_matrix.append(dict[word])
            else:
                sentence_matrix.append([0.] * 100)
        matrix = np.asarray(sentence_matrix, dtype=theano.config.floatX)
        if num % gap == 0:
            q_list.append(matrix)
        else:
            a_list.append(matrix)
        num += 1
    f.close()

    q_list = np.asarray(q_list)
    a_list = np.asarray(a_list)

    s_list = load_data_s(label_file)

    dataset = []
    dataset.append([q_list[:5181], a_list[:31086], s_list[:31086]])
    dataset.append([q_list[5181:], a_list[31086:], s_list[31086:]])

    return dataset

def load_data_s(label_file='data/train/labels.txt'):
    f = open(label_file, mode='r')
    s_list = []
    for line in f:
        s_list.append(int(line))
    s_list = np.asarray(s_list)
    f.close()
    return s_list

def load_data_qa(embedding_file='data/qas_test/test01/dict.txt',
                  word_file='data/qas_test/test01/words.txt',
                  gap=5):
    dict = load_embedding_to_dict(embedding_file)
    f = open(word_file, mode='r')

    q_list = []
    a_list = []
    num = 0
    for line in f:
        line = line.strip()
        sentence_matrix = []
        sentence_matrix.append([0.] * 100)
        for word in line.split(" "):
            if dict.has_key(word):
                sentence_matrix.append(dict[word])
            else:
                sentence_matrix.append([0.] * 100)
        matrix = np.asarray(sentence_matrix, dtype=theano.config.floatX)
        if num % gap == 0:
            q_list.append(matrix)
        else:
            a_list.append(matrix)
        num += 1
    f.close()

    q_list = np.asarray(q_list)
    a_list = np.asarray(a_list)

    return q_list, a_list

if __name__ == '__main__':
    x, y = load_data_xy(x_file='data/train/words.txt', y_file='data/train/labels.txt',
                            embedding_file='data/train/dict.txt', sample_sum=34824)
    print y[0]
