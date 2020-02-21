---
title: "Clustering Described"
date: 2017-28-03T23:40:49+00:00
description : "Machine Learning / Clustering "
type: post
image: https://scikit-learn.org/stable/_images/sphx_glr_plot_dbscan_0011.png
author: Akhil Gupta
tags: ["Clustering Analysis"]
---
After Supervised Learning algorithms, it’s time to have a look at the most popular Unsupervised method. Here, we present to you - **_Clustering_**, and it’s variants.

Let’s look at it’s simplicity [here](https://en.wikipedia.org/wiki/Cluster_analysis)

In our daily life, we group different activities according to their utility. This grouping is what you need to learn.

> **A winner is just a loser who tried one more time. Keep trying.**

---

# What is Clustering?

- How does a _recommendation system_ work?

- How does a company decide the location for their new store so as to generate\*\* **maximum **profit\*\*?

It’s an [unsupervised learning](http://www.gatsby.ucl.ac.uk/~dayan/papers/dun99b.pdf) algorithm which groups the given data, such that **data points with similar behaviour** are merged into one group.

Main aim is to segregate the various data points into different groups called _clusters_ such that entities in a particular group comparatively have more similar traits than entities in another group.
\*At the end, **_each data point is assigned to one of the group_**.

Clustering algorithm **does not** predict an outcome or target variable but can be used to improve predictive model. Predictive models can be built for clusters to _improve the accuracy of our prediction_.

![**Easy peasy.**](https://cdn-images-1.medium.com/max/964/1*rs4650azDz6FP5cRVxs2WQ.png)

---

# Types of Clustering

There exist more than 100 clustering algorithms as of today.
Some of the commonly used are **k-Means**, **Hierarchical**, DBSCAN and OPTICS. Two of these have been covered here:

#### 1. Hierarchical Clustering

It is a type of connectivity model clustering which is based on the fact that data points that are closer to each other are more similar than the data points lying far away in a data space.

As the name speaks for itself, the hierarchical clustering **forms the hierarchy of the clusters** that can be studied by visualising [_dendogram_](https://en.wikipedia.org/wiki/Dendrogram).

![**Dendogram.**](https://cdn-images-1.medium.com/max/1600/1*AplNpGVerhVpqmXf1dKd9w.png)

> **How to measure closeness of points?**

- **Euclidean distance**: ||a-b||2 = √(Σ(ai-bi))

- **Squared Euclidean distance**: ||a-b||22 = Σ((ai-bi)²)

- **Manhattan distance**: ||a-b||¹ = Σ|ai-bi|

- **Maximum distance**:||a-b||^inf = maxi|ai-bi|

- **Mahalanobis distance**: √((a-b)T S-1 (-b)) {where, s : covariance matrix}

> **How to calculate distance between two clusters?**

1. **Centroid Distance**: Euclidean distance between mean of data points in the two clusters

1. **Minimum Distance**: Euclidean distance between two data points in the two clusters that are closest to each other

1. **Maximum Distance** : Euclidean distance between two data points in the two clusters that are farthest to each other

- **_Focus on Centroid Distance right now!_**

#### Algorithm Explained

1. Let there be **_N_** data points. Firstly, these _N_ data points are assigned to _N_ different clusters with one data point in each cluster.

1. Then, two data points with **minimum euclidean distanc**e between them are merged into a single cluster.

1. Then, two clusters with **minimum centroid distance** between them are merged into a single cluster.

1. This **_process is repeated_** until we are left with a single cluster, hence forming hierarchy of clusters.

![***How it is done!***](https://cdn-images-1.medium.com/max/1632/1*ZSPU7LV3vXbdRudDTiff6Q.png)

> **How many clusters to form?**

1. **Visualising dendogram:** Best choice of no. of clusters is _no. of vertical lines that can be cut by a horizontal line_, that can transverse maximum distance vertically without intersecting other cluster.
   For eg., in the below case, best choice for no. of clusters will be **_4_**.

1. **Intuition** and prior knowledge of the data set.

![**Focus on A and B.**](https://cdn-images-1.medium.com/max/1536/1*LBOReupihNEsI6Kot3Q6YQ.png)

#### Good Cluster Analysis

- **Data-points within same cluster share similar profile**: Statistically, check the standard deviation for each input variable in each cluster. A perfect separation in case of cluster analysis is rarely achieved. Hence, even **_one standard deviation distance_** between two cluster means is considered to be a good separation.

- **Well spread proportion of data-points among clusters**: There are no standards for this requirement. But a minimum of 5% and maximum of 35% of the total population can be assumed as a safe range for each cluster.

![**Implementation in Python!**](https://cdn-images-1.medium.com/max/1720/1*sqE0Rui1GCUZ9hLeWkJjqg.jpeg)

---

#### K-Means Clustering

One of the simplest and most widely used unsupervised learning algorithm. It involves a simple way to classify the data set into fixed no. of **_K_** clusters . The idea is to define **_K_** centroids, one for each cluster.

The final clusters depend on the initial configuration of centroids. So, they should be initialized as far from each other as possible.

- K-Means is _iterative_ in nature and _easy_ to implement.

#### Algorithm Explained

- Let there be **_N_** data points. At first, **_K _**centroids are initialised in our data set representing *K *different clusters.

![**Step 1: N = 5, K = 2**](https://cdn-images-1.medium.com/max/686/1*kYXvKVSPVnw86RxgAOBSMA.png)

- Now, each of the **_N_** data points are assigned to closest centroid in the data set and merged with that centroid as a single cluster. In this way, every data point is assigned to one of the centroids.

![**Step 2: Calculating the centroid of the 2 clusters**](https://cdn-images-1.medium.com/max/690/1*Q2AHrnZ7qRsg14NDxalPVg.png)

- Then, **_K_** cluster centroids are recalculated and again, each of the **_N_** data points are assigned to the nearest centroid.

![**Step 3: Assigning all the data points to the nearest cluster centroid**](https://cdn-images-1.medium.com/max/686/1*BAp7MPVmDZ0UQWv5ZOUbuw.png)

- Step 3 is repeated until no further improvement can be made.

![**Step 4: Recalculating the cluster centroid. After this step, no more improvement can be made.**](https://cdn-images-1.medium.com/max/684/1*3t1EXtGfDtbTeLIwm8MoSQ.png)

### In this process, a loop is generated. As a result of this loop, K centroids change their location step by step until no more change is possible.

This algorithm aims at minimising the **objective function**:

![](https://cdn-images-1.medium.com/max/600/1*a_RNgesXtp0lkq5LPSWaLg.jpeg)

It represent the sum of** euclidean distance** of all the data points from the cluster centroid which is minimised.

![**Implementation in Python!**](https://cdn-images-1.medium.com/max/900/1*jrsZ4iEOmhkMciLvbliBJw.png)

> **How to initialize K centroids?**

1. **Forgy:** Randomly assigning K centroid points in our data set.

1. **Random Partition:** Assigning each data point to a cluster randomly, and then proceeding to evaluation of centroid positions of each cluster.

1. [**KMeans++]**(https://en.wikipedia.org/wiki/K-means%2B%2B#Improved_initialization_algorithm): Used for \***small\*\*\* data sets.

1. [**Canopy Clustering]**(https://en.wikipedia.org/wiki/Canopy_clustering_algorithm): Unsupervised pre-clustering algorithm used as preprocessing step for K-Means or any Hierarchical Clustering. It helps in speeding up clustering operations on \***large data sets\*\*\*.

> **How to calculate centroid of a cluster?**

Simply the mean of all the data points within that cluster.

> **How to find value of K for the dataset?**

In K-Means Clustering, value of **_K _**has to be specified beforehand. It can be determine by any of the following methods:

- **Elbow Method**: Clustering is done on a dataset for varying values of and **SSE (Sum of squared errors)** is calculated for each value of _K_.
  Then, a graph between _K_ and SSE is plotted. Plot formed assumes the shape of an arm. There is a point on the graph where SSE does not decreases significantly with increasing _K_. This is represented by elbow of the arm and is chosen as the value of _K_. (OPTIMUM)

![**Code in Python!**](https://cdn-images-1.medium.com/max/1314/1*sqjquWMTaRHBxCyM1XDhFw.png)

![**K can be 3 or 4.**](https://cdn-images-1.medium.com/max/1296/1*ZQ_7QFLnLbE3pr4_Meu0Iw.jpeg)

- [**Silhouette Score**](<https://en.wikipedia.org/wiki/Silhouette_(clustering)>): Used to study the **_separation distance_** between the resulting clusters. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighbouring clusters. C[lick here](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) for complete explanation of the method.

![**More the Algos, powerful the Arsenal.**](https://cdn-images-1.medium.com/max/2572/1*VmwXbcBS02prWvKOAn-z_A.png)

---

# K-Means v/s Hierarchical

1. For **big data**, **_K-Means_** is better!
   Time complexity of K-Means is linear, while that of hierarchical clustering is quadratic.

1. Results are reproducible in **_Hierarchical_**, and not in K-Means, as they depend on intialization of centroids.

1. K-Means requires prior and proper knowledge about the data set for specifying **_K_**. In **_Hierarchical_**, we can choose no. of clusters by interpreting dendogram.

---

#### References

1. AV’s [Blog](https://www.analyticsvidhya.com/blog/2013/11/getting-clustering-right-part-ii/) on Clustering

1. [Blog](https://iitrdsg.wordpress.com/2016/06/15/k-means-clustering-explained/) on K-Means by Akhil Gupta

1. [Python Machine Learning](https://www.amazon.in/Python-Machine-Learning-Sebastian-Raschka-ebook/dp/B00YSILNL0?_encoding=UTF8&tag=googinhydr18418-21) by Sebastian Raschka

#### Footnotes

You are getting richer day-by-day. 7 down, 5 more to go!
Start applying for internships. You can rock the interviews. Just stick to [**12A12D**](https://medium.com/data-science-group-iitr/algos-algos-everywhere-f4e684473f14).

Thanks for reading. :)
_And, ❤ if this was a good read. Enjoy!_

Co-Authors: [Nishant Raj](https://medium.com/u/1101baa51aff) and [Pranjal Khandelwal](https://medium.com/u/33c77ec08bba)
