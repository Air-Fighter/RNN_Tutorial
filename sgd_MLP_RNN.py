import time
import sys
import os
import cPickle

import theano
import theano.tensor as T
import numpy as np

from LoadData_xy import load_data_xy
from MLP_RNN import MLP

def sgd_MLP(learning_rate=0.01,
            L1_reg=0.00,
            L2_reg=0.0001,
            max_epoches=1000,
            dataset='data/RNNinput.txt',
            n_hidden=400):

    print '...reading data'
    datasets = load_data_xy()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # build model
    print '...building model'

    x = T.matrix('x', dtype=theano.config.floatX)
    y = T.vector('y', dtype='int64')

    rng = np.random.RandomState(1234)
    classifier = MLP(rng = rng,
                     input=x,
                     n_in=100,
                     n_hidden=n_hidden,
                     n_out=2)

    cost = classifier.loss_function(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    test_model = theano.function(inputs=[x, y], outputs=classifier.errors(y))

    valid_model = theano.function(inputs=[x, y], outputs=classifier.errors(y))

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

    valid_freq = 1

    best_params = None
    best_valid_loss = np.inf
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
            valid_losses = [valid_model(valid_set_x[i], [valid_set_y[i]]) for i in xrange(len(valid_set_y))]
            this_valid_loss = np.mean(valid_losses)

            print '\tepoch', epoch, ' validation error:', this_valid_loss * 100.

            if this_valid_loss < best_valid_loss:
                if this_valid_loss < best_valid_loss * improvement_threshold :
                    patience = max(patience, epoch * patience_increase)

                best_valid_loss = this_valid_loss

                file = open('data/best_params.txt', 'wb')
                cPickle.dump(classifier.params, file)
                file.close()

                test_losses = [test_model(test_set_x[i], [test_set_y[i]]) for i in xrange(len(test_set_y))]
                test_score = np.mean(test_losses)

                print '\tepoch %i, test error of best model %f %%' % \
                    (epoch,  test_score * 100.)

        if epoch >= patience:
            done_looping = True
            break

    end_time = time.clock()
    print 'Optimization comlete with best validation score of %f %%, with test performance %f %%' %\
            (best_valid_loss * 100., test_score * 100.)
    print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The ' + os.path.split(__file__)[1] +
                          ' ran for %.lfs' % ((end_time - start_time)))


if __name__ == '__main__':
    sgd_MLP()