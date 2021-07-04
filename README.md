# Time series Self Supervised Learning

The paper under scrutiny, Unsupervised Scalable Representation Learning for Multivariate Time Series, proposes an unsupervised method for learning universal embeddings of time series.


In this work we reproduced the results from the paper and attempted to improve them. We used the same loss as the SimCLR paper to outperform the method proposed by the authors. We also improved the efficiency of the negative sampling strategy. We proved that the architecture allow for good transfer capacity which allows for future work training on more data hoping to generalize more. Finally, we proved that the architecture works well on multidimensional time series which allow to unify approaches using the same embedding space for different dimensions of time series.

The Notebook Experiments contains all of the experiments.

Here are some embeddings visualisation that our method yield on 3 datasets (ECG5000, ElectricDevices and ECG5days):

<p align='center'><img src= 'ECG5000.png' width='33%'/><img src= 'ElectricDevices.png' width='33%'/><img src= 'ECG5days.png' width='33%'/></p>
