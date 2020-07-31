# Generates a tensorflow graph which runs a Discrete Cosine Transform on an 8x8
# block. It's written into a binary file called tfdct.pb
import os
import tensorflow as tf

path = os.getcwd()
tf.compat.v1.disable_eager_execution()
sess=tf.compat.v1.InteractiveSession()

# Suppose we have an 8x8 matrix x, and we want to calculate its discrete cosine
# transform y. We can use the following formula:
#                           y = T * x * (T^-1)
# T is an 8x8 matrix defined as:
# T(i, j) =
#   if i == 0 then 1 / sqrt(8)
#   else sqrt(1/4) * cos(((2 * j + 1) * i * PI) / 16)
#
# Thanks to T's orthogonality, T^-1 (the inverse of T) is equal to T's
# transpose.
# More info on https://www.math.cuhk.edu.hk/~lmlui/dct.pdf

T = tf.compat.v1.constant(
  [0.353553391, 0.353553391, 0.353553391, 0.353553391, 0.353553391, 0.353553391,
    0.353553391, 0.353553391,
   0.490392640, 0.415734806, 0.277785117, 0.097545161, -0.097545161, -0.277785117,
   -0.415734806, -0.490392640,
   0.461939766, 0.191341716, -0.191341716, -0.461939766, -0.461939766, -0.191341716,
   0.191341716, 0.461939766,
   0.415734806, -0.097545161, -0.490392640, -0.277785117, 0.277785117, 0.490392640,
   0.097545161, -0.415734806,
   0.353553391, -0.353553391, -0.353553391, 0.353553391, 0.353553391, -0.353553391,
   -0.353553391, 0.353553391,
   0.277785117, -0.490392640, 0.097545161, 0.415734806, -0.415734806, -0.097545161,
   0.490392640, -0.277785117,
   0.191341716, -0.461939766, 0.461939766, -0.191341716, -0.191341716, 0.461939766,
   -0.461939766, 0.191341716,
   0.097545161, -0.277785117, 0.415734806, -0.490392640, 0.490392640, -0.415734806,
   0.277785117, -0.097545161],
  shape=[8, 8], dtype=tf.float32)

Tinv = tf.compat.v1.transpose(T)

x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[8, 8], name="x")
inter = tf.compat.v1.linalg.matmul(T, x, name="inter")
y = tf.compat.v1.linalg.matmul(inter, Tinv, name="y")

# save the graph
tf.compat.v1.train.write_graph(graph_or_graph_def = sess.graph_def,
                               logdir = path,
                               name = "tfdct.pb",
                               as_text = False)

