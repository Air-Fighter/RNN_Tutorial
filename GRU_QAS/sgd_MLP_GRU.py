import time
import sys
import os
import cPickle
import argparse

import theano
import theano.tensor as T
import numpy as np

sys.path.append('.')

from LoadData import load_data_qas
from MLP_GRU import MLP_QAS

def sgd_MLP(
            param_file=None,
            learning_rate=0.01,
            valid_freq=5,
            L1_reg=0.00,
            L2_reg=0.0001,
            max_epoches=2000,
            dataset='data/RNNinput.txt',
            n_hidden=1000):

    print '...reading data'
    path = 'data/train/qas_train/'

    dataset = load_data_qas(
        word_file=path + 'words.txt',
        dict_file=path + 'dict.txt',
        label_file=path + 'labels.txt',
        gap=8)

    path = 'data/train/qas_train/valid/'
    validset = load_data_qas(
        word_file=path + 'words.txt',
        dict_file=path + 'dict.txt',
        label_file=path + 'labels.txt',
        gap=6
    )

    train_set_q, train_set_l, train_set_a, train_set_s = dataset
    valid_set_q, valid_set_l, valid_set_a, valid_set_s = validset


    # build model
    print '...building model'

    q = T.matrix('q', dtype=theano.config.floatX)
    l = T.matrix('l', dtype=theano.config.floatX)
    a = T.matrix('a', dtype=theano.config.floatX)
    a_ = T.matrix('a_', dtype=theano.config.floatX)

    rng = np.random.RandomState(13)
    gru = MLP_QAS(rng = rng,
                  question=q,
                  lead_in=l,
                  answer=a,
                  wanswer=a_,
                  n_in=100,
                  n_hidden=n_hidden)

    if not learning_rate is float:
        learning_rate = float(learning_rate)

    cost = gru.loss_function

    gparams = [T.grad(cost, param) for param in gru.params]

    updates = [(param, param - learning_rate * gparam)
               for param, gparam in zip(gru.params, gparams)]

    train_model = theano.function(
        inputs=[q, l, a, a_],
        outputs=cost,
        updates=updates
    )

    pred_model = theano.function(
        inputs=[q, l, a],
        outputs=gru.L2
    )


    if not param_file is None:
        print '\n...rebuilding the model from the former parameters:' + param_file + '\n'
        f= open(param_file, 'rb')
        pre_params = cPickle.load(f)
        f.close()
        gru.question_layer.U.set_value(pre_params[0])
        gru.question_layer.V.set_value(pre_params[1])
        gru.question_layer.W.set_value(pre_params[2])
        gru.lead_in_layer.U.set_value(pre_params[3])
        gru.lead_in_layer.V.set_value(pre_params[4])
        gru.lead_in_layer.W.set_value(pre_params[5])
        gru.answer_layer.U.set_value(pre_params[6])
        gru.answer_layer.V.set_value(pre_params[7])
        gru.answer_layer.W.set_value(pre_params[8])

    print "...training model"

    best_params = None
    best_valid_error = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while epoch < max_epoches:
        epoch += 1

        total_cost = 0.

        # f = open('data/gru_train/epoch_' + str(epoch) + '.txt', 'wb')
        for index in xrange(len(train_set_q)):
            # print >> f, "index:", index
            for i in xrange(3):
                this_cost = train_model(train_set_q[index], train_set_l[index],
                                        train_set_a[6 * index + 2 * i + 1], train_set_a[6*index + 2*i])
                total_cost += this_cost
                print "\repoch", epoch, " index:", index," i:", i, " cost:", this_cost,
                # print >> f, "\ti:", i, "s:", train_set_s[6 * index + i], " cost:", this_cost

        print " total loss:", total_cost
        # f.close()


        if epoch % valid_freq == 0:

            # f = open('data/gru_valid/epoch_' + str(epoch) + '.txt', 'w')
            this_valid_error = 0
            for index in xrange(len(valid_set_q)):
                # print >> f, "index:", index
                min_L2 = np.inf
                min_i = -1
                for i in xrange(4):
                    l2 = pred_model(valid_set_q[index], valid_set_a[index * 6 + i])
                    if l2 < min_L2:
                        min_L2 = l2
                        min_i = i
                    """
                    if i == 0:
                        print >> f, "q_pred:", q_pred
                        print >> f, "a_pred:", a_pred
                    else:
                        print >> f, "a_pred:", a_pred
                    """

                if min_i < 4 and min_i >= 0 and valid_set_s[index * 4 + min_i] == 1:
                    this_valid_error += 0
                else:
                    this_valid_error += 1
            # f.close()

            print '\tepoch:', epoch, ' validation error:', this_valid_error, "/", len(train_set_q)

            if this_valid_error < best_valid_error:
                best_valid_error = this_valid_error
            file = open('data/qas_params/gru_params/epoch_' + str(epoch) + '.txt', 'wb')
            cPickle.dump([param.get_value() for param in gru.params], file)
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
    parser.add_argument('-f', '-valid_frequence', help='valide frenquence when training model')
    args = parser.parse_args()

    if not args:
        sgd_MLP()
    else:
        sgd_MLP(param_file=args.p, learning_rate=args.l, valid_freq=args.f)