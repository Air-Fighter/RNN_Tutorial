import theano
import theano.tensor as T
import numpy as np
import cPickle

from MLP_RNN import MLP
from LoadData_xy import load_data_xy

def build_model_for_predication(
        data_set='data/RNNinput.txt',
        out_file='data/model_output.txt',
        n_hidden=400):
    print '...loading data'
    datasets = load_data_xy()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print '...loading parameters'

    f= open('data/best_params.txt', 'rb')
    best_params = cPickle.load(f)
    f.close()

    print '...rebuilding model'

    x = T.matrix('x', dtype=theano.config.floatX)
    rng = np.random.RandomState(1234)

    classifier = MLP(rng = rng,
                     input=x,
                     n_in=100,
                     n_hidden=n_hidden,
                     n_out=2)

    classifier.hidden_layer.U = best_params[0]
    classifier.hidden_layer.V = best_params[1]
    classifier.hidden_layer.W = best_params[2]
    classifier.output_layer.W = best_params[3]
    classifier.output_layer.b = best_params[4]

    predictor = theano.function(inputs=[x],
                                outputs=classifier.output_layer.y_pred)

    print '...printing calculate results'

    for index in xrange(len(test_set_y)):
        print predictor(test_set_x[index])[0]

if __name__ == '__main__':
    build_model_for_predication()

