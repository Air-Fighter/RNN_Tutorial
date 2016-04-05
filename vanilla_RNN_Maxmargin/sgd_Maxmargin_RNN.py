import cPickle
import os
import sys
import time
import argparse

import numpy as np
import theano
import theano.tensor as T

sys.path.append('.')

from LoadData import load_data_right_wrong
from Maxmargin_RNN import MaxMargin

def sgd_MLP(
            param_file=None,
            learning_rate=0.01,
            L1_reg=0.00,
            L2_reg=0.0001,
            max_epoches=1000,
            dataset='data/RNNinput.txt',
            n_hidden=400):

    print '...reading data'
    dataset = load_data_right_wrong('data/train/dict.txt', 'data/train/words.txt')

    train_set_right, train_set_wrong = dataset[0]
    test_set_right, test_set_wrong = dataset[1]

    # build model
    print '...building model'

    x = T.matrix('x', dtype=theano.config.floatX)
    y = T.matrix('y', dtype=theano.config.floatX)

    rng = np.random.RandomState(13)

    if not learning_rate is float:
        learning_rate = float(learning_rate)

    maxmargin = MaxMargin(rng=rng, input=[x, y], word_dim=100, hidden_dim=400)

    cost = maxmargin.loss

    gparams = [T.grad(cost, param) for param in maxmargin.params]

    updates = [(param, param + learning_rate * gparam)
                   for param, gparam in zip(maxmargin.params, gparams)]

    train_model = theano.function(
        inputs=[x, y],
        outputs=cost,
        updates=updates
    )

    valid_model = theano.function(
        inputs=[x, y],
        outputs=cost
    )

    valid_output_model = theano.function(
        inputs=[x, y],
        outputs=[maxmargin.a, maxmargin.b]
    )

    if not param_file is None:
        print '\n...rebuilding the model from the former parameters:' + param_file + '\n'
        f= open(param_file, 'rb')
        pre_params = cPickle.load(f)
        f.close()
        maxmargin.hidden_layer.U.set_value(pre_params[0])
        maxmargin.hidden_layer.V.set_value(pre_params[1])
        maxmargin.hidden_layer.W.set_value(pre_params[2])
        maxmargin.W.set_value(pre_params[3])


    print "...training model"

    valid_freq = 5

    best_valid_error = np.inf
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    while epoch < max_epoches:
        epoch += 1

        total_cost = 0.

        f = open('data/rnn_train/epoch_' + str(epoch) +'.txt', mode='w')
        for index in xrange(len(train_set_right)):
            for i in xrange(3):
                this_cost = train_model(train_set_right[index], train_set_wrong[3*index + i])
                total_cost += this_cost
                print >>f, index, i, this_cost, valid_output_model(train_set_right[index], train_set_wrong[3*index + i])
                print "\repoch", epoch, " index:", index, " cost:", this_cost,
        print " total loss:", total_cost
        f.close()

        file = open('data/rnn_params/epoch_' + str(epoch) + '.txt', 'wb')
        cPickle.dump([param.get_value() for param in maxmargin.params], file)
        file.close()

        if epoch % valid_freq == 0:
            f = open('data/rnn_valid/epoch_' + str(epoch) + '.txt', 'wb')
            this_valid_error = 0
            for index in xrange(len(test_set_right)):
                for i in xrange(3):
                    if valid_model(test_set_right[index], test_set_wrong[3 * index + i]) <= 0.1:
                        this_valid_error += 1
                    print >> f, valid_output_model(test_set_right[index], test_set_wrong[3 * index + i])

            print '\tepoch', epoch, ' validation error:', this_valid_error / len(test_set_wrong) * 100.

            if this_valid_error < best_valid_error:
                best_valid_error = this_valid_error



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