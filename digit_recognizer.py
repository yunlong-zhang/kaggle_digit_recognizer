#!/usr/bin/python3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn as sk
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import datetime

# For clearer seperation in the Terminal
print('\n--------------------------'* 2)
print(datetime.datetime.now())
print('--------------------------\n' * 2)

# def digit_convert(series):
#     out_array = []
#     for i in range(len(series)):
#         carry = [0] * 9
#         carry.insert(series[i], 1)
#         out_array.append(carry)
#     return np.array(out_array)
    
# Data extraction
# Data for model training
train_set = pd.read_csv('train.csv')
X_train_df = train_set.drop('label', axis=1).astype('float32').values.reshape(-1, 28, 28, 1)
y_train_df = train_set['label'].values.reshape(-1,1)
#y_train_df = digit_convert(y_train_df)
X_train_df = X_train_df / 255.0

# Data for model test.
Out_test_set = pd.read_csv('test.csv').values.reshape(-1, 28, 28, 1)
Out_test_set = Out_test_set / 255.0
Sample = pd.read_csv('sample_submission.csv').set_index('ImageId')

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, random_state=0)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

################## Test without Keras
# # Helper Functions
def init_weights(shape):
    init_random_dist = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)
    
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)
#  
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 
# def max_pool_2by2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 
# def convolutional_layer(input_x, shape):
#     W = init_weights(shape)
#     b = init_bias([shape[3]])
#     return tf.nn.relu(conv2d(input_x, W) + b)
# 
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b
#     
# # Placeholders
# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_true = tf.placeholder(tf.float32, shape=[None, 10])
# 
# # Layers
# convo_1 = convolutional_layer(X_train, shape=[6, 6, 1, 32])
# convo_1_pooling = max_pool_2by2(convo_1)
# convo_2 = convolutional_layer(convo_1_pooling, shape=[6, 6, 32, 64])
# convo_2_pooling = max_pool_2by2(convo_2)
# convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7 * 7 * 64])
# full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))
# hold_prob = tf.placeholder(tf.float32)
# full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)
# y_pred = normal_full_layer(full_one_dropout, 10)
# 
# # Loss Function
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=y_pred))
# 
# # Optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
# train = optimizer.minimize(cross_entropy)
# 
# # Initialize Variables
# init = tf.global_variables_initializer()
# 
# # Session
# steps = 5000
# 
# with tf.Session() as sess:
#     sess.run(init)
#     sess.run(train)
# #     for i in range(steps):
# #         batch_x, batch_y = X_train.train.batch(i * 50)
# #         sess.run(train, feed_dict={x:batch_x, y_true:batch_y, hold_prob:0.5})
# #         
# #         # Print out a message every 100 steps
# #         if i % 100 == 0:
# #             print('Currently on step {}'.format(i))
# #             print('Accuracy is: ')
# #             # Test the Train model
# #             matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
# #             
# #             acc = tf.reduce_mean(tf.cast(matches.tf.float32))
# #             
# #             print(sess.run(acc, feed_dict={x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0}))
# #             print('\n')
##########

print(tf.VERSION)
print(tf.keras.__version__)
print('\n')

input_shape = (28, 28, 1)
digit_recognizer = tf.keras.Sequential()
# Add layers
digit_recognizer.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(6,6), input_shape=input_shape))
print(digit_recognizer.output_shape)
digit_recognizer.add(tf.keras.layers.MaxPooling2D(padding='same'))
print(digit_recognizer.output_shape)
digit_recognizer.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[6,6]))
print(digit_recognizer.output_shape)
digit_recognizer.add(tf.keras.layers.MaxPooling2D(padding='same'))
print(digit_recognizer.output_shape)
digit_recognizer.add(tf.keras.layers.Flatten())
print(digit_recognizer.output_shape)
digit_recognizer.add(tf.keras.layers.Dense(128, activation='relu'))
print(digit_recognizer.output_shape)
digit_recognizer.add(tf.keras.layers.Dropout(rate=0.2))
print(digit_recognizer.output_shape)
digit_recognizer.add(tf.keras.layers.Dense(10, activation='softmax'))
print(digit_recognizer.output_shape)

optimizer = tf.keras.optimizers.Adam()
# Configuring a model
digit_recognizer.compile(optimizer=optimizer,
						loss='sparse_categorical_crossentropy',
						metrics=['accuracy'])
# Training
digit_recognizer.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate
eva = digit_recognizer.evaluate(X_test, y_test, batch_size=32)
print('Accuracy of the test fraction of the training set: ' + str(eva[1]))
# For Out_test_set
Output = digit_recognizer.predict(Out_test_set)
labels = np.argmax(Output, axis=1)
Sample['Label'] = labels
Sample.to_csv('Results.csv')