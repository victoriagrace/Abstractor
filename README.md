# RNNRhythmAbstractor

Use this recurrent neural network in a predictive setting to explore rhythm abstraction and generation. Predict the next word item in the string, using the item at time n, plus an internal representation of the state of a set of hidden units from the previous time step.

A simple RNN model built by Victoria Grace in Python using the TensorFlow to fit the rhythmic sequence prediction task. An input vector, consisting of a list of four numbers at time n, was passed through a hidden layer with 1,000 dimensions that also received input from the weights of the hidden layer from the previous time step to produce an output vector the same size as the input, that signified the networks prediction of the next input word unit at n+1 (or target). 

The flexible TensorFlow Python API allows for the assessment of different types of recurrent network cells used with TensorFlow’s core RNN methods (line 45 in multi_pred_rnn.py). Setting:
cell = tf.contrib.rnn.BasicRNNCell(hid_dim,inp_dim,activation=tf.nn.relu)
implemented the most basic RNN cells where, output = activation(W*input+U*state+B)
cell = tf.contrib.rnn.LSTMCell(hid_dim,inp_dim,activation=tf.nn.relu)
implemented long short-term memory unit RNN cells, expected to increase performance of the network. 

A Rectified Linear Unit (ReLU) activation function, defined as f(x)=max(0,x) where x in the input to the unit, was used instead of a sigmoid or tanh non-linear functions to be more computationally efficient and accelerate the convergence of stochastic gradient descent. The hidden to output layer reused the same weights for each word in a given sequence and a fully connected layer that computed each output linearly:y = w * x + b . The loss was computed using sum-of-squares differences between the predicted value (output) and true value (target = input at n+1) at each time step. 
