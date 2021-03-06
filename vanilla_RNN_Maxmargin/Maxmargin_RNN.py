import theano
import theano.tensor as T
import numpy as np

from RNN_matrix import RNN_matrix

class MaxMargin(object):
    def __init__(self, rng, input, word_dim, hidden_dim, margin=0.1):
        self.hidden_layer = RNN_matrix(
            rng=rng,
            input=input,
            word_dim=word_dim,
            hidden_dim=hidden_dim
        )

        self.W = theano.shared(
            value=rng.uniform(
                low=-np.sqrt(1. / 0.08),
                high=np.sqrt(1. / 0.08),
                size=(word_dim, )
            ).astype(theano.config.floatX),
            name = 'W',
            borrow = True
        )

        self.margin = margin

        self.param = [self.W]

        self.a = T.dot(self.W, self.hidden_layer.output[0])
        self.b = T.dot(self.W, self.hidden_layer.output[1])

        self.this_margin = self.a - self.b

        self.loss = T.maximum(0.0, self.margin - self.this_margin)

        self.L1 = abs(self.hidden_layer.U).sum() \
                  + abs(self.hidden_layer.V).sum() \
                  + abs(self.hidden_layer.W).sum()

        self.L2_sqr = (self.hidden_layer.U ** 2).sum() \
                      + (self.hidden_layer.V ** 2).sum() \
                      + (self.hidden_layer.W ** 2).sum()

        self.params = self.hidden_layer.params + self.param

