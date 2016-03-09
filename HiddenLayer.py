import theano
import theano.tensor as T
import numpy as np

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX,)
            b = theano.shared(b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        linear_out = T.dot(input, self.W) + self.b
        if activation is None:
            self.output = linear_out
        else:
            self.output = activation(linear_out)

        self.params = [self.W, self.b]