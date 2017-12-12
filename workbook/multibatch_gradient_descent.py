import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import numpy as np
import random

housing = fetch_california_housing()
m, n = housing.data.shape

scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 10000
learning_rate = 0.001
batch_size = 10
n_batches = int(np.ceil(n / batch_size)) + 1


def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indexes = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indexes]
    y_batch = housing.target.reshape(-1, 1)[indexes]
    return X_batch, y_batch


X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1., 1., seed=42), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
tf.summary.scalar(mse)

init = tf.global_variables_initializer()
data = list(zip(scaled_housing_data_plus_bias, housing.target.reshape(-1, 1)))
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter('./train', sess.graph)
    for epoch in range(n_epochs):
        total_loss = 0
        random.shuffle(data)
        for batch_index in range(n_batches):
            mini_batches = data[batch_index * batch_size: (batch_index + 1) * batch_size]
            X_batch = [i[0] for i in mini_batches]
            y_batch = [i[1] for i in mini_batches]
            _, loss = sess.run([training_op, mse], feed_dict={X: X_batch, y: y_batch})
            total_loss += loss
        train_writer.add_summary()
        if epoch % 10 == 0:
            print('loss at epoch {} = {}'.format(epoch, total_loss))
    best_theta = theta.eval()

print('best theta')
print(best_theta)
