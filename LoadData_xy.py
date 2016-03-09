import theano
import theano.tensor as T
import numpy as np

from LoadData_wordvec import load_data

def load_data_xy(x_file='data/RNNinput.txt',
              y_file='data/MLPLabels.txt'):
    x, x_ = load_data(input_file=x_file)
    Y_file = open(y_file, mode='r')
    y_ = []

    for line in Y_file:
        line = line[:-1]
        line = line.strip()
        y_.append(int(line))

    y = np.asarray(y_, dtype='int8')

    datasets = []
    datasets.append([x[:10362], y[:10362]])
    datasets.append([x[10362:11362], y[10362:11362]])
    datasets.append([x[11362:], y[11362:]])

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
    load_data_xy()
