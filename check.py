import tensorflow as tf

x = tf.placeholder(tf.float32, shape=(None, 1))
W = tf.Variable([[2.0]])
y = tf.matmul(x, W)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(y, feed_dict={x: [[3.0]]})
    print(result)
