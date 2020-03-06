import tensorflow as tf

hello = tf.constant('Hello World')

print(type(hello))

x = tf.constant(100)

print(type(x))

sess = tf.compat.v1.Session()
print(sess.run(hello))

print(sess.run(x))

#Operations
#You can line up multiple Tensorflow operations in to be run during a session:
x = tf.constant(2)
y = tf.constant(3)

with tf.compat.v1.Session() as sess:
    print('Operations with Constants')
    print('Addition',sess.run(x+y))
    print('Subtraction',sess.run(x-y))
    print('Multiplication',sess.run(x*y))
    print('Division',sess.run(x/y))

#Placeholde

x = tf.compat.v1.placeholder(tf.int64)
y = tf.compat.v1.placeholder(tf.int64)

print(x)
print(type(x))

#Defining Operations
add = tf.add(x,y)
sub = tf.subtract(x,y)
mul = tf.multiply(x,y)

d = {x:20,y:30}

with tf.compat.v1.Session() as sess:
    print('Operations with Constants')
    print('Addition',sess.run(add,feed_dict=d))
    print('Subtraction',sess.run(sub,feed_dict=d))
    print('Multiplication',sess.run(mul,feed_dict=d))

import numpy as np
# Make sure to use floats here, int64 will cause an error.
a = np.array([[5.0,5.0]])
b = np.array([[2.0],[2.0]])

print(a,", ", a.shape)
print("="*50)
print(b,", ", b.shape)
print("="*50)

mat1 = tf.constant(a)
mat2 = tf.constant(b)
#The matrix multiplication operation:
matrix_multi = tf.matmul(mat1,mat2)
#Now run the session to perform the Operation:
with tf.compat.v1.Session() as sess:
    result = sess.run(matrix_multi)
    print(result)