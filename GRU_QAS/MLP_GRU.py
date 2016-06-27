import theano
import theano.tensor as T

from GRU_matrix import GRU_matrix

class MLP_QAS(object):
    def __init__(self, rng, question, answer, signal, n_in, n_hidden):
        self.question_layer = GRU_matrix(
            rng=rng,
            input=question,
            word_dim=n_in,
            hidden_dim=n_hidden
        )

        self.answer_layer = GRU_matrix(
            rng=rng,
            input=answer,
            word_dim=n_in,
            hidden_dim=n_hidden
        )

        self.q = self.question_layer.output
        self.a = self.answer_layer.output

        self.L1 = abs(self.question_layer.U).sum() \
                  + abs(self.question_layer.V).sum() \
                  + abs(self.question_layer.W).sum() \
                  + abs(self.answer_layer.W).sum()

        self.L2_sqr = (self.question_layer.U ** 2).sum() \
                      + (self.question_layer.V ** 2).sum() \
                      + (self.question_layer.W ** 2).sum() \
                      + (self.answer_layer.W ** 2).sum()

        self.loss_function = signal * T.sqrt(T.sum((self.q - self.a) ** 2))

        self.L2 = T.sqrt(T.sum((self.q - self.a) ** 2))

        self.params = self.question_layer.params + self.answer_layer.params
