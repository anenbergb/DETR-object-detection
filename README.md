# DETR-object-detection
Implementation of DETR: End-to-End Object Detection with Transformers

# 100 Epoch Training Results
DETR requires a very long training schedule (500 epochs) to achieve the published average precision results.
The official schedule calls for 400 epochs at learning rate 1e-4 and the final 100 epochs at 1e-5.
My abreviated training run was as follows:
* 5 epoch warm-up from learning rate 3e-6 to 3e-4
* 35 epochs at learning rate 3e-4
* 60 epochs of 1-cycle cosine annealing decay from 3e-4 to 3e-6
* The learning rate of the ImageNet pretrained ResNet-50 backbone is always 10x lower than the learning rate of the DETR transformer (encoder, decoder).
<img src="https://github.com/user-attachments/assets/d61b5c45-b4fc-4759-91c5-17142cf7e942" width="600"/>

I selected a higher max learning rate of 3e-4 rather than 1e-4 because the transformer was implemented with pre-LayerNorm,
which is compatible with a higher learning rate [[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)].
Also, a higher learning rate could increase the training velocity, which would be necessary with a reduced training schedule of 100 rather than 500 epochs.
My implementation also increases the gradient clipping maximum gradient norm from 0.1 to 1.0 to increase training velocity.
The 1-cycle cosine annealing decay schedule is common in models like [ViT](https://arxiv.org/pdf/2010.11929) and pre-LN BERT language model variants.

As we can see from the following average precision results on the COCO 2017 val set, the abreviated 100 epoch training schedule significantly
underperformed the official 500 epoch training schedule from the [DETR paper](https://arxiv.org/abs/2005.12872).
The 100 epoch training run took 4 days to complete with a single Nvidia RTX 4090 GPU using a batch size of 5 
(with 12 gradient accumulation steps to achieve an effective batch size of 60, which is close to the 64 total batch size used in the offical result). 

| Model           | Epochs | AP    | AP_{50} | AP_{75} | AP_S  | AP_M | AP_L  |
|-----------------|--------|-------|---------|---------|-------|------|-------|
| DETR (official) | 500    | 42    | 62.4    | 44.2    | 20.5  | 45.8 | 61.1  |
| DETR (ours)     | 100    | 16.67 | 30.82   | 16.03   | 1.186 | 7.48 | 24.29 |

More recent papers such as [Deformable DETR](https://arxiv.org/abs/2010.04159) and [DINO](https://arxiv.org/abs/2203.03605) explain that DETR requires a 
very long training schedule to converge and achieve competitive results because attention modules processing image features are difficult to train.
The cross-attention maps must learn to attend to very sparse spatial features focusing on the object extremities,
and learn to ignore the majority of the feature map. Early attention is diffuse, and convergence to a useful attention map is slow.

Early in training, most of the predicted object queries match to "no object" background, and the loss signal is noisy. 
DETR requires many training iterations before the randomly initialized object queries begin to associate with ground truth target boxes.

DETR has difficulty detecting small objects because the transformer encoder uses only the 32x downsampled spatial features from ResNet-50, 
rather than any of the higher-resolution feature maps that would be better suited for detecting small objects.
The high-resolution feature maps would lead to an unacceptable complexity for the self-attention module in DETR's transformer encoder, 
which has a quadratic complexity with the spatial size of the input features. 

Subsequent papers such as [Deformable DETR](https://arxiv.org/abs/2010.04159) address DETR's limitations by introducing improvements such as multi-scale deformable attention.

### The training and validation loss curves for the 100 epoch training run
<img src="https://github.com/user-attachments/assets/a7f0402c-cfa0-4a1c-ad1f-a527d7ebbba2" width="1000"/>


### The results when applying a 0.75 score threshold
<img src="https://github.com/user-attachments/assets/3bf891d0-a500-4a40-baf2-d3011cec9550" width="600"/>

### The results when applying a 0.50 score threshold
<img src="https://github.com/user-attachments/assets/283ddc39-95b4-479a-b545-902f074f77c5" width="600"/>

### The results without a score threshold
<img src="https://github.com/user-attachments/assets/d05fbebe-68e3-4593-9aae-28a414de51e5" width="600"/>

### The ground truth boxes
<img src="https://github.com/user-attachments/assets/70c6c7dd-d67f-41d3-b876-a87f960ace0e" width="600"/>

##  Positional Embeddings in Transformers
Transformers, by design, are permutation-invariant. This means they don't inherently understand the order of the input sequence. For tasks like natural language processing (NLP), where word order is crucial, or object detection, where spatial relationships matter, positional embeddings are essential.

Positional embeddings inject information about the position of elements in the sequence into the model. In DETR, they're used to encode the spatial coordinates of image patches, allowing the transformer to understand the relative positions of objects.

## Sine Positional Embeddings in DETR

DETR employs sine and cosine functions to generate positional embeddings. This approach offers several advantages:
* Relative Position Encoding: Sine and cosine functions allow the model to learn relative positional relationships rather than absolute positions.
* Extrapolation: The periodic nature of sine and cosine functions enables the model to generalize to sequence lengths longer than those seen during training.
* Frequency-Based Encoding: Using different frequencies for the sine and cosine functions provides a rich representation of positional information.

How They Work
1. Coordinate Normalization:
* DETR normalizes the spatial coordinates (x, y) of each image patch to the range [0, 1].
2. Frequency Generation:
* A set of frequencies is generated using a logarithmic scale. These frequencies determine the wavelengths of the sine and cosine functions.
3. Sine and Cosine Application:
* For each coordinate (x, y) and frequency, sine and cosine values are calculated.
4. Embedding Concatenation:
* The resulting sine and cosine values are concatenated to form the positional embedding vector.
