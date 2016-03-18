import time
import sys
import os
import cPickle
import argparse

import theano
import theano.tensor as T
import numpy as np

sys.path.append('.')

from LoadData import load_data_xy
from MLP_GRU import MLP

def sgd_MLP(
            param_file=None,
            learning_rate=0.01,
            L1_reg=0.00,
            L2_reg=0.0001,
            max_epoches=2000,
            dataset='data/RNNinput.txt',
            n_hidden=400):

    print '...reading data'
    datasets = load_data_xy(x_file='data/train/words.txt', y_file='data/train/labels.txt',
                            embedding_file='data/train/dict_03171041.txt')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # build model
    print '...building model'

    x = T.matrix('x', dtype=theano.config.floatX)
    y = T.vector('y', dtype='int64')

    rng = np.random.RandomState(12)
    classifier = MLP(rng = rng,
                     input=x,
                     n_in=100,
                     n_hidden=n_hidden,
                     n_out=2)

    if not param_file is None:
        print '\n...rebuilding the model from the former parameters:' + param_file + '\n'
        f= open(param_file, 'rb')
        pre_params = cPickle.load(f)
        f.close()
        classifier.hidden_layer.U.set_value(pre_params[0])
        classifier.hidden_layer.V.set_value(pre_params[1])
        classifier.hidden_layer.W.set_value(pre_params[2])
        classifier.output_layer.W.set_value(pre_params[3])
        classifier.output_layer.b.set_value(pre_params[4])

    if not learning_rate is float:
        learning_rate = float(learning_rate)

    cost = classifier.loss_function(y) # + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    test_model = theano.function(inputs=[x, y], outputs=classifier.errors(y))

    valid_model = theano.function(inputs=[x, y], outputs=classifier.errors(y))

    y_pred_model = theano.function(inputs=[x], outputs=classifier.output_layer.y_pred)

    p_y_given_x_model = theano.function(inputs=[x], outputs=classifier.output_layer.p_y_given_x)

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [(param, param - learning_rate * gparam)
               for param, gparam in zip(classifier.params, gparams)]

    train_model = theano.function(
        inputs=[x, y],
        outputs=cost,
        updates=updates
    )



    print "...training model"

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995

    valid_freq = 5

    best_params = None
    best_valid_error = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < max_epoches) and (not done_looping):
        epoch += 1

        total_cost = 0.

        for index in xrange(len(train_set_y)):
            this_cost = train_model(train_set_x[index], [train_set_y[index]])
            total_cost += this_cost
            print "\repoch", epoch, " index:", index," y:", train_set_y[index], " cost:", this_cost,
        print " total loss:", total_cost

        if epoch % valid_freq == 0:
            valid_errors = [valid_model(valid_set_x[i], [valid_set_y[i]]) for i in xrange(len(valid_set_y))]

            f = open('data/gru_valid/epoch_' + str(epoch) + '.txt', 'wb')
            for i in xrange(len(valid_set_y)):
                print >> f, p_y_given_x_model(valid_set_x[i]), y_pred_model(valid_set_x[i]), ' ', valid_set_y[i]
            f.close()

            this_valid_error = np.mean(valid_errors)

            print '\tepoch', epoch, ' validation error:', this_valid_error * 100.

            if this_valid_error < best_valid_error:
                if this_valid_error < best_valid_error * improvement_threshold :
                    patience = max(patience, epoch * patience_increase)

                best_valid_error = this_valid_error

                file = open('data/gru_params/epoch_' + str(epoch) + '.txt', 'wb')
                cPickle.dump([param.get_value() for param in classifier.params], file)
                file.close()

                test_losses = [test_model(test_set_x[i], [test_set_y[i]]) for i in xrange(len(test_set_y))]
                test_score = np.mean(test_losses)

                print '\tepoch %i, test error of best model %f %%' % \
                    (epoch,  test_score * 100.)

        if epoch >= patience:
            done_looping = True
            break

    end_time = time.clock()
    print 'Optimization complete with best validation score of %f %%, with test performance %f %%' %\
            (best_valid_error * 100., test_score * 100.)
    print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The ' + os.path.split(__file__)[1] +
                          ' ran for %.lfs' % ((end_time - start_time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '-params', help='the path\\filename of an existing parameter file')
    parser.add_argument('-l', '-learning_rate', help='learning rate of training the model')
    args = parser.parse_args()

    if not args:
        sgd_MLP()
    else:
        sgd_MLP(param_file=args.p, learning_rate=args.l)