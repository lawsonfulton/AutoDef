import tensorflow as tf
from tensorflow_forward_ad import forward_gradients
import numpy
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

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



training_data_root = "./training_data/fixed_material_model/"
model_base_filename = training_data_root + "elu_model"

model_filename =model_base_filename + '_decoder.pb'
output_filename = model_base_filename + '_decoder_jac.pb'
with gfile.FastGFile(model_filename,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

sess = tf.Session()
input_node = sess.graph.get_tensor_by_name("decoder_input:0")
output_node = sess.graph.get_tensor_by_name("output_node0:0")

jacobians = tf_jacobian(output_node, input_node, 1)

# tf.train.write_graph(jacobians.as_graph_def(), "./", "test_jac")

# from tensorflow.python.framework import graph_util
# from tensorflow.python.framework import graph_io
# # print("pred_node_names", pred_node_names)
# constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), jacobians.name)
# graph_io.write_graph(constant_graph, output_fld, output_path, as_text=False)
# print('saved the freezed graph (ready for inference) at: ', osp.join(output_fld, output_path))
#print(sess.graph.as_graph_def())
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io    
subgraph = tf.graph_util.extract_sub_graph(sess.graph.as_graph_def(), ["decoder_input", jacobians.name[:-2]])
graph_io.write_graph(subgraph, "./", output_filename, as_text=False)
print(subgraph)
print(jacobians.name)
print(output_node.name)
