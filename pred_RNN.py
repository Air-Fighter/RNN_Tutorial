import theano
import theano.tensor as T
import numpy as np
import cPickle

from MLP_RNN import MLP
from LoadData_xy import load_data_xy

def build_model_for_predication(
        data_set='data/RNNinput.txt',
        out_file='data/model_pred/model_output.txt',
        n_hidden=400):
    print '...loading data'
    datasets = load_data_xy()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print '...loading parameters'

    f= open('data/params/epoch50_params.txt', 'rb')
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
    predictor = theano.function(inputs=[x],  outputs=classifier.output_layer.y_pred)

    classifier.hidden_layer.U.set_value(best_params[0])
    classifier.hidden_layer.V.set_value(best_params[1])
    classifier.hidden_layer.W.set_value(best_params[2])
    classifier.output_layer.W.set_value(best_params[3])
    classifier.output_layer.b.set_value(best_params[4])

    print '...printing calculate results to %s' % out_file

    f = open(out_file, 'w')
    for index in xrange(len(test_set_y)):
        print >> f, predictor(test_set_x[index])[0], test_set_y[index]
    f.close()

if __name__ == '__main__':
    build_model_for_predication()

