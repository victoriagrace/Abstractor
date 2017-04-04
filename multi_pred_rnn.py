'''
Simple RNN

Learning situation: listen to a sequence, making a prediction at each time-step

Architecture:
     out
      ^
      |
hid->hid
      ^
      |
     inp
'''

from lists import *
# from lists_test import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from array import array

import random as rd

plt.ion()

seq_len = 64 + 16
# seq_len = 64 * 2
#mb_dim = 64
# seq_len = 64 

inp_dim = 4
hid_dim = 1000
out_dim = 4
lr = 1e-4

'''setup placeholders'''
inp = []
target = []
for i in range(seq_len):
    inp.append(tf.placeholder(tf.float32,[None,inp_dim]))
    target.append(tf.placeholder(tf.float32,[None,out_dim]))
'''create computation graph'''
#the function to be applied at each timestep, change this to LSTMCell if you want
# cell = tf.contrib.rnn.BasicRNNCell(hid_dim,inp_dim,activation=tf.nn.relu)
cell = tf.contrib.rnn.LSTMCell(hid_dim,inp_dim,activation=tf.nn.relu)

#creates references to the cell function for each timestep
hid_acts,last_state = tf.contrib.rnn.static_rnn(cell,inp,dtype=tf.float32)
output = []
loss = []

#each hidden-to-output layer should use the same weights, the 'reuse' argument handles this
for i in range(seq_len):
    if i == 0:
        reuse_weights = False
    else:
        reuse_weights = True
    output.append(tf.contrib.layers.fully_connected(hid_acts[i],out_dim,scope='out',reuse=reuse_weights,activation_fn=tf.sigmoid))
    # output.append(tf.contrib.layers.fully_connected(hid_acts[i],out_dim,scope='out',reuse=reuse_weights,activation_fn=None))

    loss.append(tf.reduce_sum(tf.square(target[i]-output[i]),-1))
    #eps= 1e-10
#tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(target[i], output[i]))

#loss.append(tf.reduce_sum(-(tf.log(tf.clip_by_value(output[i],eps,1))*target[i]+tf.log(tf.clip_by_value(1-output[i],eps,1)*(1-target[i]))))) #cross entropy
#add up losses across timesteps and average over minibatch. You could average over timesteps as well, to make same lr handle different length sequences better
net_loss = tf.reduce_mean(tf.add_n(loss))
train_step = tf.train.AdamOptimizer(lr).minimize(net_loss) #Tell the optimizer what to minimize, then call it to train

'''initialize weights'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# import and format training dataset 
import json
import sys
from pprint import pprint
with open('drums_ALL.json') as data_file:
    data = json.load(data_file);

count = 0
patterns = get_data()
patterns = np.asarray(patterns)
patterns = np.transpose(patterns,[1,0,2])

print(patterns)
# import test dataset
# with open('drums_test.json') as data_file:
#     data_test = json.load(data_file);

# count = 0
# patterns_test = get_data()
# patterns_test = np.asarray(patterns_test)
# patterns_test = np.transpose(patterns_test,[1,0,2])


'''create data'''
print(patterns.shape)
my_data = {}

# my_test_data = {}

#inds = np.random.randint(25,size=5)
for i in range(seq_len):

    my_data[inp[i]] = patterns[i]
    my_data[target[i]] = patterns[(i+1) % seq_len]

    # r = rd.random() # Gives me a random number between 0 and 1

    # if r < 0.8:
    #     my_data[inp[i]] = patterns[i]
    #     my_data[target[i]] = patterns[(i+1) % seq_len]
    # else:
    #     my_test_data[inp[i]] = patterns[i]
    #     my_test_data[target[i]] = patterns[(i+1) % seq_len]

test_data = {}

'''train the network for 100 steps'''
error = []
refresh = int(1e2)

# set up subplots for intput/output


#main_fig = plt.figure(1)

f, main_ax = plt.subplots(1, sharex=True)
f, axarr = plt.subplots(8, sharex=True)

# set up fixed data constants

# kick_orig = np.array([0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1])
# snare_orig = np.array([0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0])
# hihat_orig = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
# counter_orig = np.array([0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0])

# WITH len 16 RUN UP 
# kick_orig = np.array([
# 0,0,0,0,0,0,0,0,
# 0,0,0,0,0,0,0,0,
# 0,0,0,0,0,0,0,1,
# 0,1,0,0,0,0,0,1,
# 0,0,1,0,0,0,1,0,
# 0,1,0,0,0,0,0,1,
# 0,0,0,0,0,0,0,0,
# 0,1,0,0,0,0,0,1,
# 0,0,1,0,0,0,0,0,
# 0,1,0,0,0,0,0,1
# ])
# #single seq + 16 runup
# snare_orig = np.array([
# 0,0,0,0,0,0,0,0,
# 0,0,0,0,0,0,0,0,
# 0,0,0,1,0,0,0,0,
# 0,0,0,1,0,0,0,0,
# 0,0,0,1,0,0,0,0,
# 0,0,0,1,0,0,0,0,
# 0,0,0,1,0,0,0,0,
# 0,0,0,1,0,0,0,0,
# 0,0,0,1,0,0,0,0,
# 0,0,0,1,0,0,0,0
# ])

# hihat_orig = np.array([
# 0,0,0,0,0,0,0,0,
# 0,0,0,0,0,0,0,0,
# 0,1,0,1,0,1,0,1,
# 0,1,0,1,0,1,0,1,
# 0,1,0,1,0,1,0,1,
# 0,1,0,1,0,1,0,1,
# 0,1,0,1,0,1,0,1,
# 0,1,0,1,0,1,0,1,
# 0,1,0,1,0,1,0,1,
# 0,1,0,1,0,1,0,1
# ])


# counter_orig = np.array([0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0])

# #2 seq with run up

kick_orig = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
snare_orig = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0])
hihat_orig = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
counter_orig = np.array([0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0, 0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1, 0,0.3333,0.6667, 1,0,0.3333,0.6667, 1])




# added 16 counter
# counter_orig = np.array([0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0])


# snare_orig = np.array([0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0])
# hihat_orig = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
# counter_orig = np.array([0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0])


# kick_orig = np.array([
# 0,0,0,0,0,0,0,1,
# 0,1,0,0,0,0,0,1,
# 0,0,1,0,0,0,1,0,
# 0,1,0,0,0,0,0,1,
# 0,0,0,0,0,0,0,0,
# 0,1,0,0,0,0,0,1,
# 0,0,1,0,0,0,0,0,
# 0,1,0,0,0,0,0,1
# ])

# kick_orig = np.array([0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0])

# snare_orig = np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0])

# snare_orig = np.array([
# 0,0,0,1,0,0,0,0,
# 0,0,0,1,0,0,0,0,
# 0,0,0,1,0,0,0,0,
# 0,0,0,1,0,0,0,0,
# 0,0,0,1,0,0,0,0,
# 0,0,0,1,0,0,0,0,
# 0,0,0,1,0,0,0,0,
# 0,0,0,1,0,0,0,0
# ])




# hihat_orig = np.array([
# 0,1,0,1,0,1,0,1,
# 0,1,0,1,0,1,0,1,
# 0,1,0,1,0,1,0,1,
# 0,1,0,1,0,1,0,1,
# 0,1,0,1,0,1,0,1,
# 0,1,0,1,0,1,0,1,
# 0,1,0,1,0,1,0,1,
# 0,1,0,1,0,1,0,1
# ])
# counter_orig = np.array([0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0,0.3333,  0.6667, 1,0])


for i in range(int(1e5)):
    _,cur_loss,predictions = sess.run([train_step,net_loss,output],feed_dict=my_data)
    error.append(cur_loss)
    # f, axarr = plt.subplots(8, sharex=True)


    if i % refresh == 0:
        plt.clf

        plt.figure(0).clf
        plt.figure(1).clf
        plt.figure(2).clf

        main_ax.plot(error)
        plt.show()
        #plt.imshow(predictions)
        #plot.bar plt.hist plot.imshow
        ind = np.arange(len(kick_orig))	

        bar_width = 0.35
        axarr[0].bar(ind, kick_orig, bar_width)
        axarr[2].bar(ind, snare_orig, bar_width)
        axarr[4].bar(ind, hihat_orig, bar_width)
        axarr[6].bar(ind, counter_orig, bar_width)



        plt.pause(0.01)
        print("next set")

        group_count = 0
        total_word_count = 0
        # create lists to hold outputs
        cycle_kick = []
        cycle_snare = []
        cycle_hihat= []
        cycle_counter = []

        my_file = open("out.txt", "w")
        kick_out = open("kick_out_%d.txt" % i, "w")
        snare_out = open("snare_out_%d.txt" % i, "w")
        hihat_out = open("hihat_out_%d.txt" % i, "w")
        counter_out = open("counter_out_%d.txt" % i, "w")
		# set of 64 patterns
        score = open("sets/set_%d.bin" % i, "wb")
        for group in predictions: 
            word_count = 0
            time_step = 0
            my_file.write("\n### group %d\n" % group_count)
            print("### group %d, iteration %d" % (group_count, i))
            if(group_count < seq_len): 
                for word in group:
                    my_file.write("\n\t### word %d\n" % word_count)
                    # print("\t### word %d" % word_count)
                    my_file.write("\t\tkick %d: %g\n" % (time_step, word[0]))
                    cycle_kick.append(word[0])
                    # print("\t\tkick %d: %g" % (time_step, word[0]))
                    my_file.write("\t\tsnare %d: %g\n" % (time_step, word[1]))
                    cycle_snare.append(word[1])
                    # print("\t\tsnare %d: %g" % (time_step, word[1]))
                    my_file.write("\t\thihat %d: %g\n" % (time_step, word[2]))
                    cycle_hihat.append(word[2])
                    cycle_counter.append(word[3])

                    # print("\t\thihat %d: %g" % (time_step, word[2]))
                    print("%02d: [%f\t%f\t%f\t%f]" % (time_step, word[0], word[1], word[2], word[3]))
                    data = array('f', [word[0], word[1], word[2]])
                    data.tofile(score)

                    #my_file.write(str(word))
                    time_step += 1
                    word_count += 1
                    total_word_count += 1

            group_count += 1


        axarr[1].bar(ind, np.array(cycle_kick), bar_width)
        axarr[3].bar(ind, np.array(cycle_snare), bar_width)
        axarr[5].bar(ind, np.array(cycle_hihat), bar_width)
        axarr[7].bar(ind, np.array(cycle_counter), bar_width)

        score.close()
        kick_out.write("%s\n" % cycle_kick)
        snare_out.write("%s\n" % cycle_snare)
        hihat_out.write("%s\n" % cycle_hihat)
        counter_out.write("%s\n" % cycle_counter)

        kick_out.close()
        snare_out.close()
        hihat_out.close()
        counter_out.close()


#        print("%.2f" % predictions)

