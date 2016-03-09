import numpy as np
import theano
import theano.tensor as T

from LogisticRegression import LogisticRegression
from LoadData_xy import load_data_xy

datasets = load_data_xy()

x, y = datasets[0]

print x[0][1]

X = T.vector('X', dtype=theano.config.floatX)
model = LogisticRegression(input=X, n_in=100, n_out=2)

pred = theano.function(inputs=[X],
                       outputs=model.y_pred)

print pred(x[0][1])

given = theano.function(inputs=[X],
                        outputs=model.p_y_given_x)

print given(x[0][1])

log = theano.function(inputs=[X],
                      outputs=model.log_given)

print log(x[0][1])

Y = T.vector('Y', dtype='int64')
error_cal = model.errors(Y)
error = theano.function(inputs=[X, Y],
                        outputs=error_cal)

print error(x[0][1], [y[0]])

cost = model.loss_function(Y)
cost_cal = theano.function(inputs=[X, Y],
                           outputs=cost)

print cost_cal(x[0][1], [y[0]])