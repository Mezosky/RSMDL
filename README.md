# Representational Similarity Measures for Deep Learning (RSMDL)

This repository centralizes various measures of similarity for representations, specifically tailored for comparing the weights and outputs of deep learning models. It provides tools and methods to assess and analyze similarities between different models and layers, aiding in model evaluation and research.

This repository is based on the compilation of similarity metrics presented in the survey [Klabunde, 2023](https://arxiv.org/abs/2305.06329). The study exposes various types of metrics; for the coding, the logics presented in the study have been followed, and PyTorch has been used for the implementation.

Currently, the project status can be seen in the following [table](table.md). In it, you will find the different features that have been programmed and how to call each of the functions that define them.

## How to install

To install the library you can use the next command:

```python
pip install -U deep-reps
```

## How to Use

The library is user-friendly, allowing you to directly import the desired metric and generate a set of vectors for evaluation. It is important to note that the matrices to be evaluated should have the dimensionality $R^{N \times D}$, where $N$ represents the number of instances and $D$ is the dimension of each tensor representation. When comparing two matrices $R$ and $R'$, ensure that $D=D'$; otherwise, errors will occur.

As an example, a set of outputs from two Clip models is compared using the `rsm_norm_difference` metric.

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


