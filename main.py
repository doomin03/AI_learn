import os
import tensorflow as tf

print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

a = tf.constant(10)
b = tf.constant(20)

c = a+b
d = (a+b).numpy()

print(type(c))
print(c)
print(type(d),d)

d_numpy_to_tensor = tf.convert_to_tensor(d)

print(type(d_numpy_to_tensor))
print(d_numpy_to_tensor)