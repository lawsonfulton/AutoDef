import numpy as np

def fd_jacobian(f_orig, x, eps=None, is_keras=False):
    """
    Computes the jacobian matrix of f at x with finite diffences.
    If is_keras is true, then x will be wrapped in an additional array before being passed to f.
    """
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)

    n_x = len(x)
    if is_keras:
        f = lambda x: f_orig(np.array([x])).flatten()
    else:
        f = f_orig

    jac = np.zeros([n_x, len(f(x))])
    dx = np.zeros(n_x)
    for i in range(n_x): # TODO can do this without for loop
       dx[i] = eps
       jac[i] = (f(x + dx ) - f(x - dx)) / (2.0 * eps)
       dx[i] = 0.0

    return jac.transpose()


def generate_jacobian_for_tf_model(model_input_path, jacobian_output_path):
    import tensorflow as tf
    from tensorflow_forward_ad import forward_gradients
    from tensorflow.python.platform import gfile
    from tensorflow.core.protobuf import saved_model_pb2
    from tensorflow.python.util import compat
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io    

    # Backwards
    def body(y, x, i):
        n = tf.shape(y)[0]
        loop_vars = [
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float64, size=n),
        ]
        _, jacobian = tf.while_loop(
            lambda j, _: j < n,
            lambda j, result: (j+1, result.write(j, tf.gradients(y[j], x)[0][i])),
            loop_vars)
        return jacobian.stack()

    def tf_jacobian(y, x, n):
        loop_vars = [
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float64, size=n),
        ]
        _, jacobian = tf.while_loop(
            lambda i, _: i < n,
            lambda i, result: (i+1, result.write(i, body(y[i], x, i))),
            loop_vars)
        return jacobian.stack()


    with gfile.FastGFile(model_input_path, 'rb') as f:
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
    subgraph = tf.graph_util.extract_sub_graph(sess.graph.as_graph_def(), ["decoder_input", jacobians.name[:-2]])
    graph_io.write_graph(subgraph, "./", jacobian_output_path, as_text=False)
    # print(subgraph)
    # print(jacobians.name)
    # print(output_node.name)

def generate_vjp(model_input_path, vjp_output_path):
    import tensorflow as tf
    from tensorflow_forward_ad import forward_gradients
    from tensorflow.python.platform import gfile
    from tensorflow.core.protobuf import saved_model_pb2
    from tensorflow.python.util import compat
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io    

    sess = tf.Session()
    tf.reset_default_graph()

    with gfile.FastGFile(model_input_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    sess = tf.Session()
    input_node = sess.graph.get_tensor_by_name("decoder_input:0")
    output_node = sess.graph.get_tensor_by_name("output_node0:0")

    n = output_node.get_shape()[1]
    v = tf.placeholder(tf.float64, [1, n], name="input_v")
    vjp = tf.gradients(output_node, input_node, grad_ys=v, name="vjp")[0]
    # print(vjp)
    # with tf.Session() as sess:
    #   sess.run(init_op)
    #   print sess.run(vjp,  feed_dict={x: x_val, u: u_val})
    # tf.train.write_graph(jacobians.as_graph_def(), "./", "test_jac")

    def test_vjp(tol=10e-9):
        def test_feed_fwd(x):
            return sess.run(output_node, feed_dict={input_node: [x]})[0]

        def func_vjp(x, vec):
            return sess.run(vjp, feed_dict={input_node: [x], v: [vec]})[0]

        x = np.random.normal(0.0, 0.1, input_node.shape[1])
        vjp_jac = np.array([func_vjp(x, v) for v in np.identity(output_node.shape[1])])
        fd_jac = fd_jacobian(test_feed_fwd, x, eps=0.0001)

        max_error = abs(np.max(vjp_jac - fd_jac))

        if max_error > tol:
            print("Computing vjp failed! Error of", max_error, "exceeded tolerance of", tol)
            exit()
    
    test_vjp()

    # from tensorflow.python.framework import graph_util
    # from tensorflow.python.framework import graph_io
    # # print("pred_node_names", pred_node_names)
    # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), jacobians.name)
    # graph_io.write_graph(constant_graph, output_fld, output_path, as_text=False)
    # print('saved the freezed graph (ready for inference) at: ', osp.join(output_fld, output_path))
    # print(sess.graph.as_graph_def())
    subgraph = tf.graph_util.extract_sub_graph(sess.graph.as_graph_def(), ["decoder_input", vjp.name[:-2], "input_v"])
    graph_io.write_graph(subgraph, "./", vjp_output_path, as_text=False)
    # print(subgraph)
    print(vjp.name)
    # print(output_node.name)

def generate_jvp(model_input_path, jvp_output_path):
    """WARNING THIS IS GIVING BAD VALUES"""

    import tensorflow as tf
    from tensorflow_forward_ad import forward_gradients
    from tensorflow.python.platform import gfile
    from tensorflow.core.protobuf import saved_model_pb2
    from tensorflow.python.util import compat
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io    

    def fwd_gradients(ys, xs, d_xs):
      """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
      With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
      the vector being pushed forward."""
      # v = tf.placeholder(ys.dtype, shape=ys.get_shape(), name="dummy")  # dummy variable
      v = tf.ones_like(ys, name="dummy")
      g = tf.gradients(ys, xs, grad_ys=v, name="gradients")
      return (tf.gradients(g, v, grad_ys=d_xs, name="jvp"), g)

    sess = tf.Session()
    tf.reset_default_graph()

    with gfile.FastGFile(model_input_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    sess = tf.Session()
    input_node = sess.graph.get_tensor_by_name("decoder_input:0")
    output_node = sess.graph.get_tensor_by_name("output_node0:0")


    n = input_node.get_shape()[1]
    z_v = tf.placeholder(tf.float64, [1, n], name="input_z_v")
    jvp, g = fwd_gradients(output_node, input_node, d_xs=z_v)
    jvp = jvp[0]
    g = g[0]


    def test_jvp(tol=10e-9):
        def test_feed_fwd(x):
            return sess.run(output_node, feed_dict={input_node: [x]})[0]

        def func_jvp(x, vec):
            return sess.run(jvp, feed_dict={input_node: [x], z_v: [vec]})[0]

        x = np.random.normal(0.0, 0.1, input_node.shape[1])
        jvp_jac = np.column_stack([func_jvp(x, v) for v in np.identity(input_node.shape[1])])
        fd_jac = fd_jacobian(test_feed_fwd, x, eps=0.0001)

        max_error = abs(np.max(jvp_jac - fd_jac))

        if max_error > tol:
            print("Computing jvp failed! Error of", max_error, "exceeded tolerance of", tol)
            exit()
    
    test_jvp()
    # print(jvp)
    # with tf.Session() as sess:
    #   sess.run(init_op)
    #   print sess.run(jvp,  feed_dict={x: x_val, u: u_val})
    # tf.train.write_graph(jacobians.as_graph_def(), "./", "test_jac")

    # from tensorflow.python.framework import graph_util
    # from tensorflow.python.framework import graph_io
    # # print("pred_node_names", pred_node_names)
    # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), jacobians.name)
    # graph_io.write_graph(constant_graph, output_fld, output_path, as_text=False)
    # print('saved the freezed graph (ready for inference) at: ', osp.join(output_fld, output_path))
    # print(sess.graph.as_graph_def())
    subgraph = tf.graph_util.extract_sub_graph(sess.graph.as_graph_def(), ["decoder_input", jvp.name[:-2], "input_z_v", "dummy", g.name[:-2]])
    graph_io.write_graph(subgraph, "./", jvp_output_path, as_text=False)
    # print(subgraph)
    print(jvp.name)
    # print(output_node.name)

if __name__ == "__main__":
    import sys, os

    jp_type = sys.argv[1]
    model_root = sys.argv[2]
    tf_decoder_path = os.path.join(model_root, 'tf_models/decoder.pb')

    if jp_type == 'vjp':
        generate_vjp(tf_decoder_path, os.path.join(model_root, 'tf_models/decoder_vjp.pb'))
    elif jp_type == 'jvp':
        generate_jvp(tf_decoder_path, os.path.join(model_root, 'tf_models/decoder_jvp.pb'))
    else:
        print("First parameter must be vjp or jvp.")
