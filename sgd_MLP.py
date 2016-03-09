import time
import sys
import os

import theano
import theano.tensor as T
import numpy as np

from LoadData import load_data
from MLP import MLP


def sgd_MLP(learning_rate=0.01,
            L1_reg=0.00,
            L2_reg=0.0001,
            max_epoches=1000,
            dataset='/home/shawn/PycharmProjects/RNNTutorial/data/mnist.pkl.gz',
            batch_size=20,
            n_hidden=400):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_y.eval().shape[0] / batch_size
    n_valid_batches = valid_set_y.eval().shape[0] / batch_size
    n_test_batches = test_set_y.eval().shape[0] / batch_size

    # build model
    print '...building model'

    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    rng = np.random.RandomState(1234)

    classifier = MLP(rng=rng,
                     input=x,
                     n_in=28 * 28,
                     n_hidden=n_hidden,
                     n_out=10)

    cost = classifier.loss_function(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    test_model = theano.function(inputs=[index],
                                 outputs=classifier.errors(y),
                                 givens={
                                     x: test_set_x[index * batch_size : (index + 1) * batch_size],
                                     y: test_set_y[index * batch_size : (index + 1) * batch_size]
                                 })

    valid_model = theano.function(inputs=[index],
                                  outputs=classifier.errors(y),
                                  givens={
                                     x: valid_set_x[index * batch_size : (index + 1) * batch_size],
                                     y: valid_set_y[index * batch_size : (index + 1) * batch_size]
                                 })

    gparams = [T.grad(cost, param) for param in classifier.params]


    updates = [(param, param - learning_rate * gparam)
               for param, gparam in zip(classifier.params, gparams)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x : train_set_x[index * batch_size : (index + 1) * batch_size],
            y : train_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    print "...training model"

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995

    valid_freq = min(n_train_batches, patience / 2)

    best_params = None
    best_valid_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < max_epoches) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % valid_freq == 0:
                valid_losses = [valid_model(i) for i in xrange(n_valid_batches)]
                this_valid_loss = np.mean(valid_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (epoch, minibatch_index + 1, n_train_batches, this_valid_loss * 100.)
                )

                if this_valid_loss < best_valid_loss:
                    if this_valid_loss < best_valid_loss * improvement_threshold :
                        patience = max(patience, iter * patience_increase)

                    best_valid_loss = this_valid_loss
                    best_iter = iter

                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print '\tepoch %i, minibatch %i/%i, test error of best model %f %%' % \
                        (epoch, minibatch_index + 1, n_train_batches, test_score * 100.)

            if iter >= patience:
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