import theano
import theano.tensor as T
import numpy as np

class LogisticRegression(object):
    """
    This class implements a basic classifier -- logistic regression.
    """

    def __init__(self, input, n_in, n_out):
        """
        :type input: theano.tensor.TensorType
        :param input: symbolic variable
        :type n_in: int
        :param n_in: number of input units, the dimension of the space
        :type n_out: int
        :param n_out: number of output units, the dimension of the space
        """

        # initialize with 0 the weights W as a matrix of shape(n_in, n_out)
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype = theano.config.floatX
            ),
            name = 'W',
            borrow = True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value = np.zeros(
                (n_out, ),
                dtype = theano.config.floatX
            ),
            name = 'b',
            borrow = True
        )

        # symbolic expression for computing the matrix of class-membership probabilities
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

        # parameters of modal
        self.params = [self.W, self.b]

        self.log_given = T.log(self.p_y_given_x)

    def loss_function(self, y):
        """
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the correct label
        :return: the mean of the negative log-likelihood of the prediction of this modal
        .. math::
                \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
                \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,\ell (\theta=\{W,b\}, \mathcal{D})
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        This is for the early-stopping function.
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the correct label
        :return: a float representing the number of errors in the minibatch over the total number of
                examples of the minibatch; zero one loss
        """

        # check y's dimension
        if y.ndim != self.y_pred.ndim :
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        # check y's datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()