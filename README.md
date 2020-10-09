# IIC with Contrastive Learning


## Introduction
[*Invariant Information Clustering for Unsupervised Image Classification and Segmentation*](https://arxiv.org/abs/1807.06653) (here IIC) presents a model for unsupervised clustering of images based on maximizing mutual information between correlated images. Here we propose an extension of IIC which aims to improve the performance by adding another head which maximizes a constrastive objective. The structure of this component is inspired by [*SimCLR*](https://arxiv.org/abs/2002.05709). 

## Architecture and method
The main idea of this work is to use SimCLR learned representation to improve the performance of unsupervised clustering of IIC. In order to do this, the architecture is composed by a convolutional neural network backbone, which acts as a feature extractor. On top of this, three heads receive the CNN output: an IIC overclustering head, an IIC clustering head and a SimCLR head. In each training epoch, all three heads are separately trained. The architecture is finally evaluated on the IIC clustering head, by using average accuracy and max accuracy. This method has been chosen consistently with IIC, in order to provide a better metric of performance with respect to the original work. For the same reason, the same network settings have been used.

## Results
We compare our method with IIC as well as with other unsupervised clustering methods on the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. In the following table we report some results. Our method reaches the indicated accuracy values in 500 epochs, while original IIC metrics are measured after 2000 epochs.

Method | Accuracy
------------ | -------------
Random network | 10.1
K-means | 22.9
ADC | 32.5
IIC (max) | 61.7
IIC (avg) | 57.6
Our method (max) | 76.5
Our method (avg) | 65.8

## Credits
Part of the code is inspired by the original [IIC repo](https://github.com/xu-ji/IIC) and another [SimCLR repo](https://github.com/sthalles/SimCLR). This work was part of my bachelor's degree thesis project at the University of Trento.
