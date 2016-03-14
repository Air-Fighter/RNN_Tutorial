import theano
import theano.tensor as T
import numpy as np
import cPickle

from MLP_RNN import MLP
from LoadData import load_data_x
from LoadData import load_data_y

def build_model_for_predication(
        data_set='data/RNNinput.txt',
        out_file='data/test_4/output_pred.txt',
        n_hidden=400):
    print '...loading data'

    x_set = load_data_x('data/test_3/words.txt', 'data/test_3/dict.txt')
    y_set = load_data_y('data/test_3/labels.txt')

    print '...loading parameters'

    f= open('data/params/epoch_435.txt', 'rb')
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
    #predictor = theano.function(inputs=[x],  outputs=classifier.output_layer.y_pred)
    predictor = theano.function(inputs=[x], outputs=classifier.output_layer.p_y_given_x)

    classifier.hidden_layer.U.set_value(best_params[0])
    classifier.hidden_layer.V.set_value(best_params[1])
    classifier.hidden_layer.W.set_value(best_params[2])
    classifier.output_layer.W.set_value(best_params[3])
    classifier.output_layer.b.set_value(best_params[4])

    print '...printing calculate results to %s' % out_file

    f = open(out_file, 'w')
    for index in xrange(len(x_set)):
        print >> f, predictor(x_set[index])[0][1], y_set[index]
    f.close()

if __name__ == '__main__':
    build_model_for_predication()

