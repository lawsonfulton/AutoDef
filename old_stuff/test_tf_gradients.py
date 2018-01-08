import tensorflow as tf
from tensorflow_forward_ad import forward_gradients


import numpy
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat


model_filename ='./training_data/fixed_material_model/model_decoder.pb'
#model_filename ='./training_data/fixed_material_model/decoder'
with gfile.FastGFile(model_filename,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
# print ("!!!!")
# print ("!!!!")
# print ("!!!!")
# print ("!!!!")
# print ("!!!!")
# print ("!!!!")
# print(graph_def)

sess = tf.Session()
input_node = sess.graph.get_tensor_by_name("decoder_input:0")
output_node = sess.graph.get_tensor_by_name("output_node0:0")
# Is this different from above?
# input_node2 = sess.graph.get_operation_by_name("decoder_input:0").outputs[0]
# output_node2 =sess.graph.get_operation_by_name("output_node0:0").outputs[0]
# print (input_node)
# print (input_node2)
# print (output_node)
# print (output_node2)

input_data = [[0.0, 0.0, 0.0]]
prediction = sess.run(output_node, feed_dict={input_node:input_data})
print(prediction)
print(len(prediction[0]))
exit()
# def better_fwd_gradients(ys, xs, d_xs):
#   """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
#   With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
#   the vector being pushed forward."""
#   v = tf.placeholder(ys.dtype, shape=ys.get_shape())  # dummy variable
#   g = tf.gradients(ys, xs, grad_ys=v)
#   return tf.gradients(g, v, grad_ys=d_xs)

# q = tf.placeholder(tf.float32, [1, 3])
# z_in = numpy.zeros((1,3))
# q_in = numpy.ones((1,3))
# print(z_in)
# jvp = better_fwd_gradients(output_node, input_node, q)
# val = sess.run(jvp, feed_dict={input_node:z_in, q:q_in}) # J(z) @ q
# print(val)
# print(len(val))
# exit()

# Automatic differentiation.
# x = tf.constant(1.0)
# y = tf.square(x)


# TODO see if you can do this with forward mode instead
def body(y, x, i):
    n = tf.shape(y)[0]
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, tf.gradients(y[j], x)[0][i])),
        loop_vars)
    return jacobian.stack()

def tf_jacobian(y, x, n):
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda i, _: i < n,
        lambda i, result: (i+1, result.write(i, body(y[i], x, i))),
        loop_vars)
    return jacobian.stack()

x = input_node
y = output_node

jacobians = tf_jacobian(output_node, input_node, 1)

import time
start = time.time()
val = sess.run(jacobians, feed_dict={input_node:input_data})
print("took:", time.time()-start)
print(val)
print(len(val))

for i in range(3):
    h = 0.0001
    input_data = [[0.0, 0.0, 0.0]]
    f = sess.run(output_node, feed_dict={input_node:input_data})
    input_data[0][i] = h
    fp = sess.run(output_node, feed_dict={input_node:input_data})
    print((fp - f)/h)

# OK. So the jacobian works. Now I just need to export it as a model

# exit()
# [array([[ 0.07585978, -0.02526504, -0.00470799, ...,  0.02607962,
#         -0.01195237, -0.01606977]], dtype=float32)]
# dydx = forward_gradients(y, x)

# val = sess.run(dydx, feed_dict={input_node:input_data})
# print(val)
# print(len(val))

# exit()
# # Computes Jacobian-vector product.
# x = tf.ones([5, 10])
# y = tf.square(x)
# v = tf.ones([5, 10]) * 2
# Jv = forward_gradients(y, x, v)
# sess = tf.Session()
# print(sess.run(Jv))  # [array([[ 4.,  4.,  4.,  4., ...

# # A list of inputs.
# x1 = tf.ones([5, 10])
# x2 = tf.ones([10, 8])
# y1 = tf.square(tf.matmul(x1, x2))
# y2 = tf.sigmoid(tf.matmul(x1, x2))
# v1 = tf.ones([5, 10]) * 0.5
# v2 = tf.ones([10, 8]) * 2.0
# J1v, J2v = forward_gradients([y1, y2], [x1, x2], [v1, v2])

