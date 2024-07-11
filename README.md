# Representational Similarity Measures for Deep Learning (RSMDL)

<p align="center"> 
<img src="./docs/similarity.png" width=450/>
</p>

This repository centralizes various measures of similarity for representations, specifically tailored for comparing the weights and outputs of deep learning models. It provides tools and methods to assess and analyze similarities between different models and layers, aiding in model evaluation and research.

This repository is based on the compilation of similarity metrics presented in the survey [Klabunde, 2023](https://arxiv.org/abs/2305.06329). The study exposes various types of metrics; for the coding, the logics presented in the study have been followed, and PyTorch has been used for the implementation.

Currently, the project status can be seen in the following [table](table.md). In it, you will find the different metrics that have been programmed and how to call each of the functions that define them.

## What are the Representational Similarities?

<p align="center"> 
<img src="./docs/representations.png" width=450/>
</p>

Representation similarity compares the outputs of two models given a set of inputs $X$. The goal is to compare representations $R$ and $R'$ that represent a batch of tensor representations obtained from specific layers of the models being studied. Formally, we can define a model as:

$$F := (f^{(l)} \circ f^{(l-1)} \circ \dots \circ f^{(1)})$$

Thus, a representation $R^{(l)}$ of an instance $X$ is given by:

$$R^{(l)} = f^{(l)}(X) \in \mathbb{R}^{N \times D}$$

Here, the representation of the $i$-th instance is given by a vector $R_i^{(l)} \in \mathbb{R}^{1 \times D}$. For convenience, the information representing an instance is always represented as a vector of length $D$. However, if you have tensors for $R_i^{(l)}$, it is recommended to use a projection or a flattened version of the representation.

## How to install

To install the library you can use the next command:

```python
pip install -U deep-reps
```

## How to Use

The library is user-friendly, allowing you to directly import the desired metric and generate a set of vectors for evaluation. It is important to note that the matrices to be evaluated should have the dimensionality $R^{N \times D}$, where $N$ represents the number of instances and $D$ is the dimension of each tensor representation. When comparing two matrices $R$ and $R'$, ensure that $D=D'$; otherwise, errors will occur.

As an example, a set of outputs from two Clip models are compared using the `rsm_norm_difference` metric.

```python
import torch
from pathlib import Path 

from deep_reps import rsm_norm_difference
from deep_reps import CLIPAndTokenizerLayers

pretrained = 'openai'
model_name_1 = "ViT-B-32"
model_name_2 = "RN101"
device= "cpu"

clip_model_1 = CLIPAndTokenizerLayers(model_name=model_name_1, pretrained=pretrained, device=device)
clip_model_2 = CLIPAndTokenizerLayers(model_name=model_name_2, pretrained=pretrained, device=device)

image_paths = [str(path) for path in Path("./images").glob("*")]
images_classes = ["capybara", "cat", "dog", "duck", "hedeghog", "hyrax"]

images_features_1 = []
images_features_2 = []
for path in image_paths:
    image_features_1, _ = clip_model_1.forward(path, images_classes)
    image_features_2, _ = clip_model_2.forward(path, images_classes)

    images_features_1.append(image_features_1[''])
    images_features_2.append(image_features_2[''])
images_features_1 = torch.cat(images_features_1)
images_features_2 = torch.cat(images_features_2)

print(f"RSM Norm Difference: {rsm_norm_difference(images_features_1, images_features_2)}")
>>RSM Norm Difference: 0.8515523672103882
```


