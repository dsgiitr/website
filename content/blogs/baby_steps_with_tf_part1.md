---
title: "Baby steps with Tensorflow #1"
date: 2017-06-01T23:40:49+00:00
description : "Computer Vision / Deep Learning"
type: post
image: https://cdn-images-1.medium.com/max/800/1*ZBRUL9BfVbxmyv8CGgcR_A.jpeg
author: Ajay Unagar
tags: ["Deep Learning, NLP"]
---

### Deep Learning is a Mandate for Humans, Not Just Machines — Andrew Ng

Yes, Deep learning is a [next big thing](https://www.quora.com/Is-Deep-Learning-the-next-big-thing-in-AI). Yes, AI is changing [the world](https://www.wired.com/insights/2014/07/artificial-intelligence-changing-world-humankind-must-adapt/). Yes, It will [take over your jobs](http://www.huffingtonpost.com/quora/what-jobs-sectors-will-ar_b_14498720.html). All fuss aside, you need to sit down and write a code on your own to see things working. Practical knowledge is as much important as Theoretical knowledge.

![Fact: I’m not a coder.](https://cdn-images-1.medium.com/max/1600/0*11la7wgbiwa1tKPS.png)

There are many deep learning frameworks which make it really easy and fast to train different models and deploy it. Tensorflow, PyTorch, Theano, Keras, Caffe, Tiny-dnn and list goes on. [Here](https://deeplearning4j.org/compare-dl4j-torch7-pylearn) is the great comparison for those want to know pros and cons of each one.

We will focus on Tensorflow. (For all the great debaters out there, I’m not a supporter of anyone. Turns out tensorflow is relatively new with really great resources to learn and most of the industries seems to use it.) I’m learning tensorflow from other resources, so I will try to merge them here in best way possible. This will surely help those who haven’t used Tensorflow before, I can not say anything for others. Though it is assumed that you all have basic under standing of Neural Networks, Loss functions, Optimization techniques, Backpropagation, etc. If not I suggest you go through [this great book](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen. I will also be mentioning more often other libraries like numpy, sklearn, matplotlib, etc.

#### Constants, Variables, Placeholders and Operations

**Constants**: Here constants has same meaning as in any other programming language. They stores constant value. (Integer, float, etc.)

```python
c1 = tf.constant(value = 32, dtype = tf.float32, name = 'a')
c2 = tf.constant(value = 20, dtype = tf.float32, name = 'b')
```

Where will you be using constants? Value which are not supposed to change! Like number of layers, shape of the weight vectors, shape of each layer, etc. Some of the great constant initializers are [here](https://www.tensorflow.org/api_guides/python/constant_op) (most like numpy). Thing to notice is that you can not even get a value of tensor until you initialize a session. What is session? We will get to it.

**Variables**:

Variables are those, which will be updated in Tensorflow graph. For example: Weights and biases. More about variables [here](https://www.tensorflow.org/programmers_guide/variables).

```python
# variable 1 with initial value 100
v1 = tf.Variable(initial_value=[100], name = 'v1')

# variable 3 with initial value initialized by a constant
v3 = tf.Variable(initial_value= tf.random_normal(shape= [100 , 4], mean= 0.0, stddev= 1), name = 'v3')
```

**Placeholders**:

Placeholders are as name suggest reserve space for the data. So, while feed forwarding you can feed data into network through placeholders. Placeholders have defined shape. If your input data has n-dimensions, you need to specify n-1 dimensions and then while feeding, you can feed data into batches in the network. More about placeholders [here](https://www.tensorflow.org/versions/r0.11/api_docs/python/io_ops/placeholders).

```python
# For example lets say you have MNIST data
# MNIST digits are 28*28 pixels, so you need to specify this
#dimension in placeholder.
#Here shape = [None, 28*28], where None can take any value.
ph1 = tf.placeholder(dtype= tf.float32, shape= [None, 28*28])
```

**Operations:**

Operation are basic function we define on variables, constants and placeholders. More about operations [here](https://www.tensorflow.org/api_guides/python/math_ops)

```python
# multiply c1 and c2
o1 = tf.multiply(c1, c2)

# sum of elements of v3
o4 =tf.reduce_sum(v3)
```

#### Session

Session is a class implemented in Tensorflow for running operations, and evaluating constants and variables. More about session [here](https://www.tensorflow.org/api_docs/python/tf/Session).

```python
# Start a session
sess = tf.Session()

# We can evaludate o1 and o2
print(sess.run(o1), sess.run(o2))

# We need to initialize all the variables defined
# before evaluating operations which contain variables
sess.run(tf.global_variables_initializer())

# Now we can evaluate operation on variables
print(sess.run(o4))
```

#### How Tensoflow works?

What are the main steps of any machine learning algorithm in Tensorflow?

1. Import the data, normalize it, or create data input pipeline.

1. Define an algorithm — Define variables, structure of the algo, loss function, optimization technique, etc. Tensorflow creates static computational graphs for this.

1. Feed the data through this computation graph, compute loss from loss function and and update the weights (variables) by backpropagating the error.

1. Stop when you reach some stopping criteria.

Here is vary simple example for multiply operation on two constants.

![](https://cdn-images-1.medium.com/max/1456/0*uvXAUUtje1B01o_s.png)

And here is other simple computation graph.

![Simple computation Graph](https://cdn-images-1.medium.com/max/1082/1*UUJDu2UBCDv0tWJuW19syQ.png)

That’s it for now!

#### Source Code:

You can find source code for this assignment on my [github repo.](https://github.com/ajayunagar/Baby-Steps-With-Tensorflow/tree/master/Tensorflow%20Basics)

# References:

1. Really good and detailed [blog](https://medium.com/@camrongodbout/tensorflow-in-a-nutshell-part-one-basics-3f4403709c9d).

1. [Github](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction)

1. [Getting Started with Tensorflow](https://www.tensorflow.org/get_started/get_started)

Next we will see Linear and Logistic Regression.

Hit ❤ if you find this useful. :D
