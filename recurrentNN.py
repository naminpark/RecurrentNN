'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''



class RecurrentNN():

    batch_size = 128
    display_step = 10
    training_iters = 100000

    def __init__(self,lr=0.001,n_input =28,n_steps=28,n_hidden=128,n_classes=10,weight='he',bias='zero'):
        # Parameters
        self.learning_rate = lr

        # Network Parameters
        self.n_input = n_input # MNIST data input (img shape: 28*28)
        self.n_steps = n_steps # timesteps
        self.n_hidden = n_hidden # hidden layer num of features
        self.n_classes = n_classes # MNIST total classes (0-9 digits)
        self.setVariable()

        self.weights = self.weight_initializer[weight]
        self.biases = self.bias_initializer[bias]

        self.Model()
        init = tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)

    def setVariable(self):
        # tf Graph input
        self.x = tf.placeholder("float", [None, self.n_steps, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])

        # Define weights
        self.normal_weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }

        self.xavier_weights = {
            'out': tf.get_variable('out_xaiver',[self.n_hidden, self.n_classes],initializer=tf.contrib.layers.xavier_initializer())
        }

        self.he_weights = {
            'out': tf.get_variable('out_he',[self.n_hidden, self.n_classes],initializer=tf.contrib.layers.variance_scaling_initializer())
        }


        self.normal_biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        self.zero_biases = {
            'out': tf.Variable(tf.zeros([self.n_classes]))
        }

        self.weight_initializer = {'normal':self.normal_weights, 'xavier':self.xavier_weights, 'he':self.he_weights}
        self.bias_initializer = {'normal':self.normal_biases, 'zero':self.zero_biases}



    def Model(self):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        x = tf.transpose(self.x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, self.n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, self.n_steps, 0)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        pred= tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']



        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    def RUN(self,mnist):

        step = 1
        # Keep training until reach max iterations
        while step * self.batch_size < self.training_iters:
            batch_x, batch_y = mnist.train.next_batch(self.batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((self.batch_size, self.n_steps, self.n_input))
            # Run optimization op (backprop)
            self.sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})
            if step % self.display_step == 0:
                # Calculate batch accuracy
                acc = self.sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y})
                # Calculate batch loss
                loss = self.sess.run(self.cost, feed_dict={self.x: batch_x, self.y: batch_y})
                print("Iter " + str(step*self.batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for 128 mnist test images
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, self.n_steps, self.n_input))
        test_label = mnist.test.labels[:test_len]
        print("Testing Accuracy:", \
            self.sess.run(self.accuracy, feed_dict={self.x: test_data, self.y: test_label}))




if __name__=="__main__":

    RNN=RecurrentNN()

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    RNN.RUN(mnist)