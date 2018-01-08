import numpy as np
import numpy.random as npr
import tensorflow as tf

def fwd_gradients(ys, xs, d_xs):
  """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
  With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
  the vector being pushed forward."""
  v = tf.placeholder(ys.dtype, shape=ys.get_shape())  # dummy variable
  g = tf.gradients(ys, xs, grad_ys=v)
  return tf.gradients(g, v, grad_ys=d_xs)

A = tf.constant(npr.randn(5, 3), dtype=tf.float32)
x = tf.placeholder(tf.float32, [1, 5])
y = tf.tanh(tf.matmul(x, A))
u = tf.placeholder(tf.float32, [1, 5])

jvp = fwd_gradients(y, x, u)

x_val = npr.randn(1, 5)
u_val = npr.randn(1, 5)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
  sess.run(init_op)
  print(sess.run(jvp,  feed_dict={x: x_val, u: u_val}))