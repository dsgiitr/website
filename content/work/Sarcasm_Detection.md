---
title: "Sarcasm Detection using BERT"
type: work
date: 2019-12-02T23:40:49+00:00
description : "This is meta description"
caption: Implementation of BERT based Sarcasm Detection Classification model using Tensorflow.
image: https://img.buzzfeed.com/buzzfeed-static/static/2016-03/10/14/enhanced/webdr01/enhanced-11415-1457636668-6.png?output-quality=auto&output-format=auto&downsize=640
author: Dhruv Kalsotra
tags: ["NLP", "Deep Learning", "BERT"]
submitDate: December 28, 2019
github: https://github.com/dsgiitr/Sarcasm-Detection-Tensorflow
---

# What Is Sarcasm?
Sarcasm, a sharp and ironic utterance designed to cut or to cause pain, is often used to express strong
emotions, such as contempt, mockery or bitterness. 

## Why?

In order to correctly understand people’s true intention, being able to detect sarcasm is critical. Many previous models have been developed to detect sarcasm based on the utterances in isolation, meaning only the reply text itself. The main models used were SVMs
and LSTMs with attention. In this project, we aim to improve accuracy of sarcasm detection by further exploring the role of contextual information in detecting sarcasm. We will investigate the performances of different models including LSTM models and BERT. 


Sarcasm detection is of great importance in understanding people’s true sentiments and opinions. Application of sarcasm detection can benefit many areas of interest of NLP applications, including marketing research, opinion mining and information categorization. However, sarcasm detection is also a very difficult task, as it’s largely dependent on context, prior knowledge and the tone in which the sentence was spoken or written.

## Dataset

News Headlines Dataset.

Each record consists of :

```is_sarcastic: 1 if the record is sarcastic otherwise 0```

```headline: the headline of the news article```

```article_link: link to the original news article. Useful in collecting supplementary data```

Pre-trained BERT and LSTM models in tensorflow used as the model takes in the headlines as input and, output the class.