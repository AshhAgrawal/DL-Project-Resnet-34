# ResNet-34 Inspired Lightweight Model for CIFAR-10 Classification

## Project Overview
This project presents a lightweight variant of ResNet-34 optimized for CIFAR-10 image classification. The model is designed to achieve a balance between computational efficiency and classification accuracy, making it suitable for deployment in resource-constrained environments.

## Repository
[GitHub Repository](https://github.com/AshhAgrawal/DL-Project-Resnet-34)

## Model Architecture
The model is built upon ResNet-34 but incorporates several optimizations:
- **Residual Learning:** Uses residual blocks to facilitate gradient flow.
- **BasicBlock-Based Design:** Employs a simpler block structure with two 3×3 convolutional layers, each followed by Batch Normalization and ReLU activation.
- **Optimized Channel Configuration:**
  - First Layer: 48 channels
  - Second Layer: 96 channels
  - Third Layer: 192 channels
  - Fourth Layer: 384 channels
- **Global Average Pooling:** Reduces dimensionality before classification.
- **Dropout Regularization:** Applied at 0.5 to mitigate overfitting.
- **Skip Connections:** Enhances training stability and gradient propagation.

## Dataset
The model is trained on the **CIFAR-10** dataset:
- **60,000 color images (32×32 pixels) categorized into 10 classes**
- **Training Set:** 50,000 images
- **Test Set:** 10,000 images

### Data Augmentation Techniques
To enhance generalization, the following augmentation strategies were applied:
- **Random Horizontal Flip**
- **Random Cropping**
- **Cutout**
- **AutoAugment**

## Training Details
- **Optimizer:** Stochastic Gradient Descent (SGD) with Nesterov Momentum (0.9)
- **Batch Size:** 64
- **Epochs:** 200 (pruned model trained for 50 epochs)
- **Learning Rate Schedule:** Cosine Annealing
- **L2 Regularization:** 5×10⁻⁴
- **Label Smoothing:** 0.1

### Learning Rate Scheduling (Cosine Annealing Formula)
```math
\eta_t = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos \left(\frac{t}{T} \pi \right)\right)
```
where **T** is the total number of epochs.

## Model Pruning
To enhance efficiency, structured pruning was applied to remove redundant channels and layers while maintaining high accuracy.

## Results
- The **Narrow ResNet-34 model achieved 86.36% accuracy**, outperforming the standard version with fewer parameters.
- **Pruning reduced model complexity while retaining competitive accuracy (84.75%)**.
- Strong regularization techniques prevented overfitting and improved generalization.

## Future Work
- Further model pruning and quantization.
- Experimenting with knowledge distillation techniques.
- Generalization analysis on CIFAR-100 and Tiny ImageNet.

## References
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/pdf/1805.09501)
- [Structured Pruning for Deep Convolutional Networks](https://arxiv.org/abs/2303.00566)

