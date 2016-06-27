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
            valid_freq=5,
            max_epoches=1000,
            n_hidden=400,
            n_samples=34824):

    print '...reading data'

    if not n_samples is int:
        n_samples = int(n_samples)

    path = 'data/train/5803train-m/'

    datasets = load_data_xy(x_file=path + 'words.txt', y_file=path + 'labels.txt',
                            embedding_file=path + 'dict.txt', sample_sum=n_samples)


    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    print "Training example num:", len(train_set_y)
    print "Valid example num:", len(valid_set_y)

    # build model
    print '...building model'

    x = T.matrix('x', dtype=theano.config.floatX)
    y = T.vector('y', dtype='int64')

    rng = np.random.RandomState(13)
    classifier = MLP(rng = rng,
                     input=x,
                     n_in=100,
                     n_hidden=n_hidden,
                     n_out=2)

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

    train_model = theano.function(inputs=[x, y], outputs=cost, updates=updates)


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


    if not valid_freq is int:
        valid_freq = int(valid_freq)


    print "...training model"

    best_params = None
    best_valid_error = np.inf
    test_score = 0.
    start_time = time.clock()

    epoch = 0

    while epoch < max_epoches:
        epoch += 1

        total_cost = 0.

        f = open('data/gru_train/epoch_' + str(epoch) + '.txt', 'wb')
        for index in xrange(len(train_set_y)):
            this_cost = train_model(train_set_x[index], [train_set_y[index]])
            total_cost += this_cost
            print >> f, p_y_given_x_model(train_set_x[index])[0], train_set_y[index]
            print "\repoch", epoch, " index:", index," y:", train_set_y[index], " cost:", this_cost,
        print " total loss:", total_cost
        f.close()


        if epoch % valid_freq == 0:

            valid_errors = [valid_model(valid_set_x[i], [valid_set_y[i]]) for i in xrange(len(valid_set_y))]

            this_valid_error = np.mean(valid_errors)

            print '\tepoch', epoch, ' validation error:', this_valid_error * 100.,

            right_num = 0
            for index in xrange(0, len(valid_set_y) / 4):
                max_p = 0
                max_i = -1
                for i in xrange(0, 4):
                    p = p_y_given_x_model(valid_set_x[4 * index + i])[0][1]
                    if p > max_p:
                        max_p = p
                        max_i = i
                if valid_set_y[4*index + max_i] == 1:
                    right_num += 1

            print ' right ratio:', (right_num / len(valid_set_y))

            file = open('data/5803params/gru_params/epoch_' + str(epoch) + '.txt', 'wb')
            cPickle.dump([param.get_value() for param in classifier.params], file)
            file.close()

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
    parser.add_argument('-f', '-frequency', help='frequency of validation')
    parser.add_argument('-n', '-number_of_samples', help='the number of training samples')
    args = parser.parse_args()

    if not args:
        sgd_MLP()
    else:
        sgd_MLP(param_file=args.p, learning_rate=args.l, valid_freq=args.f, n_samples=args.n)