import theano
import theano.tensor as T
import numpy as np

import operator

class RNN_matrix:

    def __init__(self, rng, input, word_dim, hidden_dim=100, bptt_truncate=4):

        # Assign instance variables
        self.U = theano.shared(
            value=rng.uniform(
                low=-np.sqrt(1./word_dim),
                high=np.sqrt(1./word_dim),
                size=(hidden_dim, word_dim)
            ).astype(theano.config.floatX),
            name='U'
        )

        self.V = theano.shared(
            value=rng.uniform(
                low=-np.sqrt(1./hidden_dim),
                high=np.sqrt(1./hidden_dim),
                size=(word_dim, hidden_dim)
            ).astype(theano.config.floatX),
            name='V'
        )

        self.W = theano.shared(
            value=rng.uniform(
                low=-np.sqrt(1./hidden_dim),
                high=np.sqrt(1./hidden_dim),
                size=(hidden_dim, hidden_dim)
            ).astype(theano.config.floatX),
            name='W'
        )

        self.params = [self.U, self.V, self.W]

        def one_step(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(T.dot(U, x_t) + T.dot(W, s_t_prev))
            o_t = T.nnet.softmax(T.dot(V, s_t))
            return o_t[0], s_t

        [o, _], updates = theano.scan(
            fn=one_step,
            sequences=[input],
            outputs_info=[None, dict(initial=T.zeros(hidden_dim), dtype=theano.config.floatX)],
            non_sequences=[self.U, self.V, self.W],
            truncate_gradient=bptt_truncate,
            strict=True
        )

        self.output_sequence = o
        self.output = o[-1]

    """
    def loss_function(self, Y):
        return abs(T.sum(T.nnet.categorical_crossentropy(coding_dist=self.output_sequence, true_dist=Y)))
    """

# the following is useless
def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)