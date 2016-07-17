import theano
import theano.tensor as T

from GRU_matrix import GRU_matrix

class MLP_QAS(object):
    def __init__(self, rng, question, lead_in, answer, wanswer, n_in, n_hidden):
        self.question_layer = GRU_matrix(
            rng=rng,
            input=question,
            word_dim=n_in,
            hidden_dim=n_hidden
        )

        self.lead_in_layer = GRU_matrix(
            rng=rng,
            input=lead_in,
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
        self.l = self.lead_in_layer.output
        self.a = self.answer_layer.output
        self.a_ = self.answer_layer.compute_answer(wanswer)[-1]

        self.L1 = abs(self.question_layer.U).sum() \
                  + abs(self.question_layer.V).sum() \
                  + abs(self.question_layer.W).sum() \
                  + abs(self.answer_layer.W).sum()

        self.L2_sqr = (self.question_layer.U ** 2).sum() \
                      + (self.question_layer.V ** 2).sum() \
                      + (self.question_layer.W ** 2).sum() \
                      + (self.answer_layer.W ** 2).sum()

        # max-margin loss function
        self.loss_function = T.maximum(0,
                                       T.sqrt(T.sum((self.q + self.l - self.a) ** 2)) - 0.1 - T.sqrt(
                                           T.sum((self.q + self.l - self.a_) ** 2)))

        self.L2 = T.sqrt(T.sum((self.q + self.l - self.a) ** 2))

        self.params = self.question_layer.params + self.lead_in_layer.params + self.answer_layer.params
