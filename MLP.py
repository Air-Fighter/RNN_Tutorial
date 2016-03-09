import theano
import theano.tensor as T
import numpy as np

from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hidden_layer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
        )

        self.output_layer = LogisticRegression(
            input=self.hidden_layer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.output_layer.W).sum()
        self.L2_sqr = (self.hidden_layer.W ** 2).sum() + (self.output_layer.W ** 2).sum()

        self.loss_function = self.output_layer.loss_function
        self.errors = self.output_layer.errors

        self.params = self.hidden_layer.params + self.output_layer.params
