---
title: "GraphSAGE (SAmple and aggreGatE) : Inductive Learning on Graphs"
date: 2020-01-01T23:40:49+00:00
description : "Machine Learning / Graph Representation Learning"
type: post
image: images/blogs/GraphSAGE/GraphSAGE_cover.jpg
author: Ajit Pant, Shubham Chandel, Shashank Gupta, Anirudh Dagar
tags: ["Graph Representation Learning"]
---

<h1><center>Introduction</center></h1>

In the previous blogs, we covered GCN and DeepWalk, which are methods to generate node embeddings. The basic
idea behind node embedding approaches is to use dimensionality reduction techniques to distill the
high-dimensional information about a node’s neighborhood into a dense vector embedding. These
node embeddings can then be fed to downstream machine learning systems and aid in tasks such as
node classification, clustering, and link prediction. Let us move on to a slightly different problem. Now, we need the embeddings for each node of a graph where new nodes are continuously being added. A possible way to do this would be to rerun the entire model (GCN or DeepWalk) on the new graph, but it is computationally expensive. Today we will be covering GraphSAGE, a method that will allow us to get embeddings for such graphs is a much easier way. Unlike embedding approaches that are based on matrix factorization, GraphSAGE leverage node features (e.g., text attributes, node profile information, node degrees) in order to learn an embedding function that generalizes to unseen nodes.


GraphSAGE is capable of learning
structural information about a node’s role in a graph, despite the fact that it is inherently based on
features

<hr/>

<h1><center>The Start</center></h1>
In the (GCN or DeepWalk) model, the graph was fixed beforehand, let's say the 'Zachary karate club', some model was trained on it, and then we could make predictions about a person X, if he/she went to a particular part of the club after separation.

<center><img src="/images/blogs/GraphSAGE/karate_club.png" width=800x/></center>
<center><strong>Zachary Karate Club</strong></center>


In this problem, the nodes in this graph were fixed from the beginning, and all the predictions were also to be made on these fixed nodes. In contrast to this, take an example where 'Youtube' videos are the nodes and assume there is an edge between related videos, and say we need to classify these videos depending on the content. If we take the same model as in the previous dataset, we can classify all these videos, but whenever a new video is added to 'YouTube', we will have to retrain the model on the entire new dataset again to classify it. This is not feasible as there will be too many videos or nodes being added everytime for us to retrain.

To solve this issue, what we can do is not to learn embeddings for each node but to learn a function which, given the features and edges joining this node, will give the embeddings for the node. 

<hr/>

<h1><center>Aggregating Neighbours</center></h1>

The idea is to generate embeddings, based on the neighbourhood of a given node. In other words, the embedding of a node will depend upon the embedding of the nodes it is connected to. Like in the graph below, the node 1 and 2 are likely to be more similar than node 1 and 5.

<center><img src="/images/blogs/GraphSAGE/example_graph_1.png"></center>
<center><strong>Simple_graph</strong></center>

How can this idea be formulated?

First, we assign random values to the embeddings, and on each step, we will set the value of the embedding as the average of embeddings for all the nodes it is directly connected. The following example shows the working on a simple linear graph.

<center><img src="/images/blogs/GraphSAGE/animation.gif"></center>
<center><strong>Mean_Embeddings</strong></center>

This is a straightforward idea, which can be generalized by representing it in the following way,

<img src="/images/blogs/GraphSAGE/simple_neighbours.png">
<center><strong>Simple Neighbours</strong></center>

Here The Black Box joining A with B, C, D represents some function of the A, B, C, D. ( In the above animation, it was the mean function). We can replace this box by any function like say sum or max. This function is known as the aggregator function.

Now let's try to make it more general by using not only the neighbours of a node but also the neighbours of the neighbours. The first question is how to make use of neighbours of neighbours. The way which we will be using here is to first generate each node's embedding in the first step by using only its neighbours just like we did above, and then in the second step, we will use these embeddings to generate the new embeddings. Take a look at the following 

<img src="/images/blogs/GraphSAGE/aggregation_1.png">
<center><strong>One_Layer_Aggregation</strong></center>

The numbers written along with the nodes are the value of embedding at the time, T=0.

Values of embedding after one step are as follows:

<img src="/images/blogs/GraphSAGE/animation_2_bw.gif">
<center><strong>Aggregation Layer 1</strong></center>

So after one iteration, the values are as follows:

<img src="/images/blogs/GraphSAGE/aggregation_2.png">
<center><strong>Aggregation After One Layer</strong></center>>

Repeating the same procedure on this new graph, we get (try verifying yourself)

<img src="/images/blogs/GraphSAGE/aggregation_3.png">
<center><strong>Aggregation After Two Layer</strong></center>

Lets try to do some analysis of the aggregation. Represent by $A^{(0)}$ the initial value of embedding of A(i.e. 0.1), by $A^{(1)}$ the value after one layer(i.e. 0.25) similarly $A^{(2)}$, $B^{(0)}$, $B^{(1)}$ and all other values.

Clearly 

$$A^{(1)} = \frac{(A^{(0)} + B^{(0)} + C^{(0)} + D^{(0)})}{4}$$

Similarly

$$A^{(2)} = \frac{(A^{(1)} + B^{(1)} + C^{(1)} + D^{(1)})}{4}$$

Writing all the value in the RHS in terms of initial values of embeddings we get

$$A^{(2)} = \frac{\frac{(A^{(0)} + B^{(0)} + C^{(0)} + D^{(0)})}{4} + \frac{A^{(0)}+B^{(0)}+C^{(0)}}{3} + \frac{A^{(0)}+B^{(0)}+C^{(0)}+E^{(0)} +F^{(0)}}{5} + \frac{A^{(0)}+D^{(0)}}{2}}{4}$$

If you look closely, you will see that all the nodes that were either neighbour of A or neighbour of some neighbour of A are present in this term. It is equivalent to saying that all nodes that have a distance of less than or equal to 2 edges from A are influencing this term. Had there been a node G connected only to node F. then it is clearly at a distance of 3 from A and hence won't be influencing this term.

Generalizing this we can say that if we repeat this produce N times, then all the nodes ( and only those nodes) that are at a within a distance N from the node will be influencing the value of the terms.

If we replace the mean function, with some other function, lets say $F$, then, in this case, we can write,

$$A^{(1)} = F(A^{(0)} , B^{(0)} , C^{(0)} , D^{(0)})$$

Or more generally

$$A^{(k)} = F(A^{(k-1)} , B^{(k-1)} , C^{(k-1)} , D^{(k-1)})$$

If we denote by $N(v)$ the set of neighbours of $v$, so $N(A)=\{B, C, D\}$ and $N(A)^{(k)}=\{B^{(k)}, C^{(k)}, D^{(k)}\}$, the above equation can be simplified as

$$A^{(k)} = F(A^{(k-1)}, N(A)^{(k-1)} )$$

This process can be visualized as:

<img src="/images/blogs/GraphSAGE/showing_1.png">
<center><strong>Showing one</strong></center>

This method is quite effective in generating node embeddings. But there is an issue if a new node is added to the graph how can get its embeddings? This is an issue that cannot be tackled with this type of model. Clearly, something new is needed, but what? 

One alternative that we can try is to replace the function F by multiple functions such that in the first layer it is 
F1, in second layer F2 and so on, and then fixing the number of layers that we want, let's say k.

So our embedding generator would be like this,
<img src="/images/blogs/GraphSAGE/showing_2.png">

Let's formalize our notation a bit now so that it is easy to understand things.

1. Instead of writing $A^{(k)}$  we will be writing $h_{A}^{k}$
2. Rename the functions $F1$, $F2$ and so on as, $AGGREGATE_{1}$, $AGGREGATE_{2}$ and so on. i.e, $Fk$ becomes $AGGREGATE_{k}$.
3. There are a total of $K$ aggregation functions.
3. Let our graph be represented by $G(V, E)$ where $V$ is the set of vertices and $E$ is the set of edges.

## What GraphSAGE proposes?

What we have been doing by now can be written as 

Initialise($h_{v}^{0}$) $\forall v \in V$ <br>
for $k=1..K$ do <br>
&nbsp;&nbsp;&nbsp;&nbsp;for $v\in V$ do<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$h_{v}^{k}=AGGREGATE_{k}(h_{v}^{k-1}, \{h_{u}^{k-1} \forall u \in N(v)\})$

$h_{v}^{k}$ will now be containing the embeddings

### Some issues with this

Please take a look at the sample graph that we discussed above, in this graph even though the initial embeddings for $E$ and $F$ were different, but because their neighbours were same they ended with the same embedding, this is not a good thing as there must be at least some difference between their embeddings. 

GraphSAGE proposes an interesting idea to deal with it. Rather than passing both of them into the same aggregating function, what we will do is to pass into aggregating function only the neighbours and then concatenating this vector with the vector of that node. This can be written as:

$h_{v}^{k}=CONCAT(h_{v}^{k-1},AGGREGATE_{k}( \{h_{u}^{k-1} \forall u \in N(v)\}))$

In this way, we can prevent two vectors from attaining exactly the same embedding.

Lets now add some non-linearity to make it more expressive. So it becomes

$h_{v}^{k}=\sigma[W^{(k)}.CONCAT(h_{v}^{k-1},AGGREGATE_{k}( \{h_{u}^{k-1} \forall u \in N(v)\}))]$

Where \sigma is some non-linear function (e.g. RELU, sigmoid, etc.) and $W^{(k)}$ is the weight matrix, each layer will have one such matrix. If you looked closely, you would have seen that there no trainable parameters till now in our model. The $W$ matrix has been added to have something that the model can learn.

One more thing we will add is to normalize the value of h after each iteration, i.e., divide them by their L2 norm, and hence our complete algorithm becomes.

<img src="/images/blogs/GraphSAGE/graphsage_algorithm.png">
<center><strong>GraphSAGE_Algorithm</strong></center>


To get the model learning, we need the loss function. For the general unsupervised learning problem, the following loss problem serves pretty well.

<img src="/images/blogs/GraphSAGE/Loss_function.png">
<center><strong>Loss Function</strong></center>

This graph-based loss function encourages nearby nodes to have similar representations, while enforcing
that the representations of disparate nodes are highly distinct.


For supervised learning, either we can learn the embeddings first and then use those embeddings for the downstream task or combine both the part of learning embeddings and the part of applying these embeddings in the task into a single end to end models and then use the loss for the final part, and backpropagate to learn the embeddings while solving the task simultaneously.

# Aggregator Architectures
One of the critical difference between GCN and Graphsage is the generalisation of the aggregation function, which was the mean aggregator in GCN. So rather than only taking the average, we use generalised aggregation function in GraphSAGE. GraphSAGE owes its inductivity to its aggregator functions.

## Mean aggregator 
Mean aggregator is as simple as you thought it would be. In mean aggregator we simply
take the elementwise mean of the vectors in **{h<sub>u</sub><sup>k-1</sup>     ∀u ∈ N (v)}**.
In other words, we can average embeddings of all nodes in the neighbourhood to construct the neighbourhood embedding.
<img src="/images/blogs/GraphSAGE/ma.png">?
<center><strong>Mean Aggregator</strong></center>


## Pool aggregator


Until now, we were using a weighted average type of approach. But we could also use pooling type of approach; for example, we can do elementwise min or max pooling. So this would be another option where we are taking the messages from our neighbours, transforming them and applying some pooling technique(max-pooling or min pooling).
<img src="/images/blogs/GraphSAGE/pa.png">
<center><strong>Pool Aggregator</strong></center>

In the above equation, max denotes the elementwise max operator, and σ is a nonlinear activation function (yes you are right it can be ReLU). Please note that the function applied before the max-pooling can be an arbitrarily deep multi-layer perceptron, but in the original paper, simple single-layer architectures are preferred.

## LSTM aggregator
We could also use a deep neural network like LSTM to learn how to aggregate the neighbours. Order invariance is important in the aggregator function, but since LSTM is not order invariant,  we would have to train the LSTM over several random orderings or permutation of neighbours to make sure that this will learn that order is not essential.

# Inductive capability

One interesting property of GraphSAGE is that we can train our model on one subset of the graph and apply this model on another subset of this graph. The reason we can do this is that we can do parameter sharing, i.e. those processing boxes are the same everywhere (W and B are shared across all the computational graphs or architectures). So when a new architecture comes into play, we can borrow the parameters (W and B), do a forward pass, and we get our prediction. 
<img src="/images/blogs/GraphSAGE/sharing_param.png">
<center><strong>Sharing Parameters</strong></center>

This property of GraphSAGE is advantageous in the prediction of protein interaction. For example, we can train our model on protein interaction graph from model organism A (left-hand side in the figure below) and generate embedding on newly collected data from other model organism say B (right-hand side in the figure).
<img src="/images/blogs/GraphSAGE/protein.png">
<center><strong>Protein Interaction</strong></center>

We know that our old methods like DeepWalk were not able to generalise to a new unseen graph. So if any new node gets added to the graph, we had to train our model from scratch, but since our new method is generalised to the unseen graphs, so to predict the embeddings of the new node we have to make the computational graph of the new node, transfer the parameters to the unseen part of the graph and we can make predictions.

<img src="/images/blogs/GraphSAGE/new_node.png">
<center><strong>new node</strong></center>

We can use this property in social-network (like Facebook). Consider the first graph in the above figure, users in a social-network are represented by the nodes of the graph. Initially, we would train our model on this graph. After some time suppose another user is added in the network, now we don't have to train our model from scratch on the second graph, we will create the computational graph of the new node, borrow the parameters from the already trained model and then we can find the embeddings of the newly added user.

# Implementation in PyTorch

## Imports


```python
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
```

## GraphSAGE class


```python
"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes,
                    [self.adj_lists[int(node)] for node in nodes], self.num_sample)
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return combined


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())
```

## Load and Run


```python
def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("./cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = [float(x) for x in info[1:-1]]
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("./cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_cora():
    np.random.seed(1)
    random.seed(1)
    
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data),
                                   requires_grad=False)

    agg1 = MeanAggregator(features, cuda=True)
    
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True,
                          cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(),
                          cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(),
                   enc1.embed_dim, 128, adj_lists, agg2, 
                   base_model=enc1, gcn=True, cuda=False)
    
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad,
                                       graphsage.parameters()), lr=0.7)
    times = []
    
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print (batch, loss.item())

    val_output = graphsage.forward(val)
    
    print ("Validation F1:", f1_score(labels[val],
                            val_output.data.numpy().argmax(axis=1),
                            average="micro"))
    
    print ("Average batch time:", np.mean(times))


if __name__ == "__main__":
    run_cora()
```

    /home/solsec/anaconda3/envs/GCN/lib/python3.7/site-packages/ipykernel_launcher.py:84: UserWarning: nn.init.xavier_uniform is now deprecated in favor of nn.init.xavier_uniform_.
    /home/solsec/anaconda3/envs/GCN/lib/python3.7/site-packages/ipykernel_launcher.py:113: UserWarning: nn.init.xavier_uniform is now deprecated in favor of nn.init.xavier_uniform_.


    0 1.942228078842163
    1 1.921658992767334
    2 1.9006750583648682
    3 1.873147964477539
    4 1.833079218864441
    5 1.793070912361145
    6 1.7698112726211548
    7 1.7396035194396973
    8 1.6929861307144165
    9 1.6441305875778198
    10 1.5536351203918457
    11 1.5488044023513794
    12 1.4822677373886108
    13 1.468451738357544
    14 1.3974864482879639
    15 1.3166505098342896
    16 1.2732900381088257
    17 1.195784330368042
    18 1.0451050996780396
    19 0.9867343306541443
    20 0.9533907175064087
    21 0.9308909177780151
    22 0.8159271478652954
    23 0.7914730906486511
    24 0.7673667669296265
    25 0.7801153063774109
    26 0.677147626876831
    27 0.6584917902946472
    28 0.6916540861129761
    29 0.7556794881820679
    30 0.7246103882789612
    31 1.0994600057601929
    32 0.8346526622772217
    33 1.0626455545425415
    34 0.5540371537208557
    35 0.4707820415496826
    36 0.47333627939224243
    37 0.4838956296443939
    38 0.4711683988571167
    39 0.4963235855102539
    40 0.48719295859336853
    41 0.4026302695274353
    42 0.35586124658584595
    43 0.4207482933998108
    44 0.41222259402275085
    45 0.3622773289680481
    46 0.33898842334747314
    47 0.3108625113964081
    48 0.34005632996559143
    49 0.38214144110679626
    50 0.314105749130249
    51 0.3763721287250519
    52 0.33562469482421875
    53 0.40695565938949585
    54 0.29900142550468445
    55 0.36123421788215637
    56 0.3518748879432678
    57 0.3004622459411621
    58 0.31813153624534607
    59 0.25553494691848755
    60 0.30214229226112366
    61 0.30288413166999817
    62 0.35318124294281006
    63 0.2550695240497589
    64 0.24285988509655
    65 0.2586570382118225
    66 0.27572184801101685
    67 0.30874624848365784
    68 0.25411731004714966
    69 0.24063177406787872
    70 0.2535572648048401
    71 0.19541779160499573
    72 0.20859725773334503
    73 0.1995910108089447
    74 0.20250269770622253
    75 0.2077709287405014
    76 0.20552675426006317
    77 0.19936150312423706
    78 0.24609258770942688
    79 0.1969422698020935
    80 0.19751787185668945
    81 0.20629757642745972
    82 0.19819925725460052
    83 0.20762889087200165
    84 0.17974525690078735
    85 0.16918545961380005
    86 0.2033073604106903
    87 0.11312698572874069
    88 0.19385862350463867
    89 0.19625785946846008
    90 0.20826341211795807
    91 0.18184316158294678
    92 0.17827709019184113
    93 0.19169804453849792
    94 0.1731080412864685
    95 0.18547363579273224
    96 0.13688258826732635
    97 0.1454528272151947
    98 0.18186761438846588
    99 0.1714990884065628
    Validation F1: 0.842
    Average batch time: 0.04003107070922852


## References
- https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
- [Graph Node Embedding Algorithms (Stanford - Fall 2019) by Jure Leskovec](https://www.youtube.com/watch?v=7JELX6DiUxQ)
- [Jure Leskovec: "Large-scale Graph Representation Learning"](https://www.youtube.com/watch?v=oQL4E1gK3VU)
- [Jure Leskovec "Deep Learning on Graphs"
](https://www.youtube.com/watch?v=MIAbDNAxChI)
