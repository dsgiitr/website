---
title: "Graph Representation Learning"
type: work
date: 2020-01-01T23:40:49+00:00
description : "This is meta description"
caption: Implementation and Explanation Graph Representation Learning papers involving DeepWalk, GCN, GraphSAGE, ChebNet and GAT.
image: images/work/graph_nets.svg
author: Subham Chandel, Ajit Pant, Shashank Gupta, Anirudh Dagar
tags: ["Graph Representation Learning", "PyTorch"]
submitDate: January 1, 2020
github: https://github.com/dsgiitr/graph_nets
---

<hr/>

This project involves a simplified, yet exhaustive approach to implementation and explanation of various Graph Representation Learning techniques developed in the recent past. We cover major papers in the field as part of the review series and we aim to add blogs on many more significant papers in the field.

<br/>

<h2 align="center"> 1. Understanding DeepWalk </h2>
<br/>
<img align="right" width="500x" height="100x" src="https://miro.medium.com/max/4005/1*j-P55wBp5PP9oqrxDxdDpw.png" style="padding-left: 20px">

Unsupervised online learning approach, inspired from word2vec in NLP, but, here the goal is to generate node embeddings.
- [DeepWalk Blog](https://dsgiitr.com/blogs/deepwalk)
- [Jupyter Notebook](https://github.com/dsgiitr/graph_nets/blob/master/DeepWalk/DeepWalk_Blog%2BCode.ipynb)
- [Code](https://github.com/dsgiitr/graph_nets/blob/master/DeepWalk/DeepWalk.py)
- [Paper -> DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)

<br/>
<h2 align="center"> 2. A Review : Graph Convolutional Networks (GCN) </h2>
<br/>
<img align="right" width="500x" src="/images/work/gcn_architecture.png" style="padding-left: 20px">

GCNs draw on the idea of Convolution Neural Networks re-defining them for the non-euclidean data domain. They are  convolutional, because filter parameters are typically shared over all locations in the graph unlike typical GNNs. 
- [GCN Blog](https://dsgiitr.com/blogs/gcn)
- [Jupyter Notebook](https://github.com/dsgiitr/graph_nets/blob/master/GCN/GCN_Blog%2BCode.ipynb)
- [Code](https://github.com/dsgiitr/graph_nets/blob/master/GCN/GCN.py)
- [Paper -> Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

<br/>
<h2 align="center"> 3. Graph SAGE(SAmple and aggreGatE) </h2>
<br/>
<img align="right" width="500x" src="/images/work/GraphSAGE_cover.jpg" style="padding-left: 20px">

Previous approaches are transductive and don't naturally generalize to unseen nodes. GraphSAGE is an inductive framework leveraging node feature information to efficiently generate node embeddings.
- [GraphSAGE Blog](https://dsgiitr.com/blogs/graphsage)
- [Jupyter Notebook](https://github.com/dsgiitr/graph_nets/blob/master/GraphSAGE/GraphSAGE_Code%2BBlog.ipynb)
- [Code](https://github.com/dsgiitr/graph_nets/blob/master/GraphSAGE/GraphSAGE.py)
- [Paper -> Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)

<br/>
<h2 align="center"> 4. ChebNet: CNN on Graphs with Fast Localized Spectral Filtering </h2>
<br/>
<img align="right" width="600x" src="https://storage.googleapis.com/groundai-web-prod/media/users/user_3036/project_14426/images/x1.png" style="padding-left: 20px">

ChebNet is a formulation of CNNs in the context of spectral graph theory.
- [ChebNet Blog](https://dsgiitr.com/blogs/chebnet/)
- [Jupyter Notebook](https://github.com/dsgiitr/graph_nets/blob/master/ChebNet/Chebnet_Blog%2BCode.ipynb)
- [Code](https://github.com/dsgiitr/graph_nets/blob/master/ChebNet/coarsening.py)
- [Paper -> Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375)

<br/>

<h2 align="center"> 5. Understanding Graph Attention Networks </h2>
<br/>
<img align="right" width="500x" src="/images/work/GAT_Cover.jpg" style="padding-left: 20px">

GAT is able to attend over their neighborhoods’ features, implicitly specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation or depending on knowing the graph structure upfront.
- [GAT Blog](https://dsgiitr.com/blogs/gat)
- [Jupyter Notebook](https://github.com/dsgiitr/graph_nets/blob/master/GAT/GAT_Blog%2BCode.ipynb)
- [Code](https://github.com/dsgiitr/graph_nets/blob/master/GAT/GAT_PyG.py)
- [Paper -> Graph Attention Networks](https://arxiv.org/abs/1710.10903)

