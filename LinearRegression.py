# coding: utf-8
import tensorflow as tf
step = 20
rows = 100
slope = .4
bias = 1.5
x_train = tf.random.uniform(shape=(rows, ))
perturb = tf.random.normal(shape=(len(x_train),), stddev=0.01)

y_train = slope * x_train + bias + perturb

y_train
m = tf.Variable(0.)
b = tf.Variable(0.)
def predict_y_value(x):
    y = m * x + b
    return y
    
def squared_error(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))
    
loss = squared_error(predict_y_value(x_train), y_train)

learning_rate = 0.05
steps = 200
for i in range(steps):
    with tf.GradientTape() as tape:
        predictions = predict_y_value(x_train)
        loss = squared_error(predictions, y_train)

for i in range(steps):
    with tf.GradientTape() as tape:
        predictions = predict_y_value(x_train)
        loss = squared_error(predictions, y_train)
    gradients = tape.gradient(loss, [m, b])
    m.assign_sub(gradients[0] * learning_rate)
    b.assign_sub(gradients[1] * learning_rate)
    if(i % step) == 0:
        print("Steps %d, Loss %f" %(i, loss.numpy()))

print("m: %.5f, b: %.5f" %(m.numpy(), b.numpy()))
print("m: %.9f, b: %.9f" %(m.numpy(), b.numpy()))
import matplotlib.pyplot as plt


y = m*x_train + b
plt.scatter(x_train, y_train, c='k', marker='o')
plt.plot(x_train, y, c='b')
plt.show()
