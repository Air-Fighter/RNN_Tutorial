import cPickle
import sys

import numpy as np
import theano
import theano.tensor as T

sys.path.append('.')

from LoadData import load_data_qa
from LoadData import load_data_s
from MLP_GRU import MLP_QAS


def build_model_for_predication(n_hidden=400):
    print '...loading data'

    test_path = '../data/qas_test/test05/'
    q_set, a_set = load_data_qa(test_path+'dict.txt', test_path+'words.txt')
    s_set = load_data_s(test_path+'labels.txt')
    out_file = test_path + 'output.txt'

    print '...loading parameters'

    f= open('../data/qas_params/epoch_187.txt', 'rb')
    best_params = cPickle.load(f)
    f.close()

    print '...rebuilding model'

    q = T.matrix('q', dtype=theano.config.floatX)
    a = T.matrix('a', dtype=theano.config.floatX)
    s = T.scalar('s', dtype='int64')
    rng = np.random.RandomState(1234)

    classifier = MLP_QAS(rng = rng,
                     question=q,
                     answer=a,
                     signal=s,
                     n_in=100,
                     n_hidden=n_hidden)

    predictor = theano.function(inputs=[q, a], outputs=classifier.L2)

    classifier.question_layer.U.set_value(best_params[0])
    classifier.question_layer.V.set_value(best_params[1])
    classifier.question_layer.W.set_value(best_params[2])
    classifier.answer_layer.U.set_value(best_params[3])
    classifier.answer_layer.V.set_value(best_params[4])
    classifier.answer_layer.W.set_value(best_params[5])

    print '...printing calculate results to %s' % out_file

    f = open(out_file, 'w')
    for index in xrange(len(q_set)):
        for i in xrange(4):
            print >> f, predictor(q_set[index], a_set[4 * index + i]), s_set[4 * index + i]
    f.close()

    ########################################################################################
    f = open(test_path+'output.txt', mode='r')

    propabilities = []
    right_one = []

    for line in f:
        propabilities.append(float(line.split(" ")[0]))
        right_one.append(int(line.split(" ")[1]))

    f.close()

    size = len(propabilities) / 4

    right_num = 0
    wrong_num = 0

    for index in xrange(size):
        right_index = -1
        min_index = -1
        min_p = 1

        for i in xrange(4):
            if propabilities[index * 4 + i] < min_p:
                min_index = i
                min_p = propabilities[index * 4 + i]
            if right_one[index * 4 + i] == 1:
                right_index = i

        if (right_index == min_index):
            right_num += 1
        else:
            wrong_num += 1

    print right_num, wrong_num
    print float(right_num) / float(right_num + wrong_num)

if __name__ == '__main__':
    build_model_for_predication()

