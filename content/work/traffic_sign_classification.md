---
title: "Beginner Friendly Project: Traffic Sign Classification"
type: work
date: 2017-11-21T23:40:49+00:00
description : "This is meta description"
caption: Using GTSRB dataset to build a CNN based Traffic Sign Classifier.
image: https://miro.medium.com/max/2400/1*PoVAzAk7lTiWXHIgm2dLeQ.jpeg
author: Utkarsh Gupta
tags: ["Deep Learning", "CNN"]
submitDate: November 21, 2017
github: https://github.com/dsgiitr/Traffic-Sign-Classification
---

<hr/>

This project involves tackling a beginner friendly computer vision based Deep Learning problem based on the dataset [The German Traffic Sign Recognition Benchmark(GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news). The problem is to to recognize the traffic sign from the images. Solving this problem is essential for self-driving cars to operate on roads.

The dataset features 43 different signs under various sizes, lighting conditions, occlusions and is very similar to real-life data. Training set includes about 39000 images while test set has around 12000 images. Images are not guaranteed to be of fixed dimensions and the sign is not necessarily centered in each image.


<hr/>
<center><b>Steps Involved</b></center>

* <b>Pre-Process Data</b> — Although deep neural networks don’t require a lot of pre-processing, but it can really accelerate the training process. We will be using a variety of transformations such as Mean, Centering, Normalizing, Histogram Equalizing Data.

* <b>Augmention</b> — Deep neural networks simply perform better when there is more data. We will be using a variety of techniques such as rotation, scaling, translation, projective transformation, sobel edge , gaussian noise to increase the robustness and variety in our dataset.

* <b>Build a model</b> — As mentioned before we will be using a convolutional neural network. The depth of convolutions, the number of layers, the type of activation functions will be chosen based on the problem at hand.

* <b>Hyper-parameter Tuning</b> — These are the knobs you need to turn around such as learning rate, number of iterations, batch size, number of epochs, dropout rate, regularization factor etc. Choosing a subset of the entire problem and building a random search over a range of hyperparameters for finding the best possible combination is a good way of proceeding.

<hr/>