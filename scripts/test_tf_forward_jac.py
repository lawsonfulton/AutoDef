# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(weights, dtype=tf.float64)

def elu(x):
    return tf.where(x >= 0.0, x, tf.exp(x) - 1)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.softplus(tf.matmul(X, w_1))  # The \sigma function
    # h    = elu(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = datasoftplus

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

# def fd_jacobian(f_orig, x, eps=None, is_keras=False):
#     """
#     Computes the jacobian matrix of f at x with finite diffences.
#     If is_keras is true, then x will be wrapped in an additional array before being passed to f.
#     """
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

def fwd_gradients(ys, xs, d_xs):
    """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
    With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
    the vector being pushed forward."""
    # dummy = tf.placeholder(ys.dtype, shape=ys.get_shape(), name="dummy")  # dummy variable
    dummy = tf.ones_like(ys)
    g = tf.gradients(ys, xs, grad_ys=dummy, name="gradients")
    return tf.gradients(g, dummy, grad_ys=d_xs, name="jvp")

def main():
    print(tf.__version__)
    train_X, test_X, train_y, test_y = get_iris_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 256                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder(tf.float64, shape=[None, x_size])
    y = tf.placeholder(tf.float64, shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    def func(x):
        return sess.run(yhat, feed_dict={X: [x]})[0]

    Vy = tf.placeholder(tf.float64, shape=[1, y_size])
    vjp = tf.gradients(yhat, X, grad_ys=Vy)

    def func_vjp(x, v):
        return sess.run(vjp, feed_dict={X: [x], Vy: [v]})[0][0]

    Vx  = tf.placeholder(tf.float64, shape=[1, x_size])
    jvp = fwd_gradients(yhat, X, d_xs=Vx)

    def func_jvp(x, v):
        return sess.run(jvp, feed_dict={X: [x], Vx: [v]})[0][0]
    
    x = np.ones(x_size)/2
    actual_jac_vjp = np.array([func_vjp(x, v) for v in np.identity(y_size)])
    print(actual_jac_vjp)

    actual_jac_jvp = np.column_stack([func_jvp(x, v) for v in np.identity(x_size)])
    print(actual_jac_jvp)

    print(np.linalg.norm(actual_jac_jvp - actual_jac_vjp))



    
    print(func(x))
    # print(np.linalg.norm(fd_jacobian(func, x, eps=0.0000001) - actual_jac_vjp))
    # print(np.linalg.norm(fd_jacobian(func, x, eps=0.000001) - actual_jac_vjp))
    # print(np.linalg.norm(fd_jacobian(func, x, eps=0.00001) - actual_jac_vjp))
    print(np.linalg.norm(fd_jacobian(func, x, eps=0.0001) - actual_jac_vjp)) # best for float 64
    # print(np.linalg.norm(fd_jacobian(func, x, eps=0.001) - actual_jac_vjp))
    # print(np.linalg.norm(fd_jacobian(func, x, eps=0.01) - actual_jac_vjp))
    # print(np.linalg.norm(fd_jacobian(func, x, eps=0.1) - actual_jac_vjp)) # best for float32
    # print(np.linalg.norm(fd_jacobian(func, x, eps=1.0) - actual_jac_vjp))

    


    # for epoch in range(100):
    #     # Train with each example
    #     for i in range(len(train_X)):
    #         sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

    #     train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
    #                              sess.run(predict, feed_dict={X: train_X, y: train_y}))
    #     test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
    #                              sess.run(predict, feed_dict={X: test_X, y: test_y}))

    #     print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
    #           % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()

if __name__ == '__main__':
    main()