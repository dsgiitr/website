---
title: "Adversarial Lab"
type: work
date: 2019-12-01T23:40:49+00:00
description : "This is meta description"
caption: This project builds on a demo for several Adversarial Attacks on ImageNet Classifier Models.
image: images/work/adversarial_example.gif
author: Saswat Das
tags: ["Deep Learning", "PyTorch", "Adversarial ML"]
submitDate: Present
github: https://github.com/dsgiitr/adversarial_lab
---
### Adversarial Lab

This project is a Web-based Tool for visualisation and generation of adversarial examples by attacking ImageNet Models like VGG, AlexNet, ResNet etc.

Visualizing and Comparision of Various Adversarial Attacks on user uploaded images using a simple interface, using the DNN framework Pytorch, using popular SOTA Pretrained TorchVision ModelZoo. The Following Attacks have been implemented so far:

1. FGSM
	* Fast Gradient Sign Method, Untargeted
	* Fast Gradient Sign Method, Targeted

2. Iterative
	* Basic Iterative Method, Untargeted
	* Least Likely Class Iterative Method

3. DeepFool, untargeted

4. LBFGS, targeted

Coming Soon: Carlini-Wagner l2, and Many More