from RNN_matrix import RNN_matrix
from LogisticRegression import LogisticRegression

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hidden_layer = RNN_matrix(
            rng=rng,
            input=input,
            word_dim=n_in,
            hidden_dim=n_hidden
        )

        self.output_layer = LogisticRegression(
            input=self.hidden_layer.output,
            n_in=n_in,
            n_out=n_out
        )

        self.L1 = abs(self.hidden_layer.U).sum() \
                  + abs(self.hidden_layer.V).sum() \
                  + abs(self.hidden_layer.W).sum() \
                  + abs(self.output_layer.W).sum()

        self.L2_sqr = (self.hidden_layer.U ** 2).sum() \
                      + (self.hidden_layer.V ** 2).sum() \
                      + (self.hidden_layer.W ** 2).sum() \
                      + (self.output_layer.W ** 2).sum()

        self.loss_function = self.output_layer.loss_function
        self.errors = self.output_layer.errors

        self.params = self.hidden_layer.params + self.output_layer.params
