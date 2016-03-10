import theano
import theano.tensor as T

from LoadData_wordvec import load_data
from RNN_matrix import RNN_matrix

X_train, Y_train = load_data(embedding_file='data/RNN_dict.txt',
                                    input_file='data/RNNinput.txt')

x = T.matrix('X', dtype=theano.config.floatX)
y = T.matrix('Y', dtype=theano.config.floatX)

model = RNN_matrix(x, word_dim=100, hidden_dim=80)

cost = model.loss_function(y)

gparams = [T.grad(cost, param) for param in model.params]

learning_rate = T.scalar('learning_rate')

updates = [(param, param - learning_rate * gparam)
           for param, gparam in zip(model.params, gparams)]


sgd = theano.function(inputs=[x, y, learning_rate], outputs=cost, updates=updates)
drive = theano.function([x, y], gparams)
out = theano.function([x], model.output)

# sgd = theano.function(inputs=[x], outputs=model.output)

for epoch in xrange(100):
    total_cost = 0
    for i in xrange(len(Y_train)):
        this_cost = sgd(X_train[i], Y_train[i], 0.0025)
        this_out = out(X_train[i])
        print "\rindex:", i, " cost:", this_cost, " out:", this_out
        # print drive(X_train[i], Y_train[i])
        total_cost += this_cost
    total_cost /= len(Y_train)
    print "epoch:", epoch, " cost:", total_cost

