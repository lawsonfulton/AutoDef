import tensorflow as tf
import numpy as np

def fwd_gradients(ys, xs, d_xs):
    dummy = tf.zeros_like(ys)
    g = tf.gradients(ys, xs, grad_ys=dummy, name="gradients")
    return tf.gradients(g, dummy, grad_ys=d_xs, name="jvp")

def my_elu(x):
    return tf.where(x >= 0.0, x, tf.exp(x) - 1.0)

def main():
    print(tf.__version__)

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    
    # activation = my_elu # Works correctly tf.nn.relu (or any other non-elu activation)
    activation = tf.nn.elu
    # activation = tf.nn.softplus

    x_size = 3
    y_size = x_size

    # Single ELU or RELU op
    X = tf.placeholder(tf.float64, shape=[x_size]) # Input
    Y = activation(X) # Output

    # Define vjp and jvp
    Vx = tf.placeholder(tf.float64, shape=[x_size]) # Jac-vector product input V
    Vy = tf.placeholder(tf.float64, shape=[y_size]) # vector-jac product input V
    jvp = fwd_gradients(Y, X, d_xs=Vx)
    vjp = tf.gradients(Y, X, grad_ys=Vy)

    # Compute jacobians
    x = np.ones(x_size) - 1.5 # Bug only occurs in x < 0 region
    # x = np.random.normal(-1, 1, x_size)
    tf_jac, numeric_jac = tf.test.compute_gradient(X, [x_size], Y, [y_size], x_init_value=x)
    vjp_jac = np.array([sess.run(vjp, feed_dict={X: x, Vy: v})[0] for v in np.identity(y_size)])
    jvp_jac = np.array([sess.run(jvp, feed_dict={X: x, Vx: v})[0] for v in np.identity(x_size)])

    # Print results as maximum absolute error
    print("Numeric jac:", numeric_jac)
    print("jvp jac:", jvp_jac)
    print("tf error:", np.max(np.abs(numeric_jac - tf_jac)))   # ~0.0
    print("vjp error:", np.max(np.abs(numeric_jac - vjp_jac))) # ~0.0
    print("jvp error:", np.max(np.abs(numeric_jac - jvp_jac))) # LARGE! for ELU

    sess.close()

if __name__ == '__main__':
    main()