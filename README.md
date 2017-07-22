# CNNs on the GTSRB
A short exploration of CNN architectures/papers for German Traffic Sign Recognition Benchmark [(GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)

## Baseline Architectures

* **LeNet5**: LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.
* **VGG16**: Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

## Papers

* Sermanet, Pierre, and Yann LeCun. "Traffic sign recognition with multi-scale convolutional networks." Neural Networks (IJCNN), The 2011 International Joint Conference on. IEEE, 2011.
* Ciresan, Dan, et al. "Multi-column deep neural network for traffic sign classification." Neural Networks 32 (2012): 333-338.

## Implementation Details

Each pipeline was implemented using OpenCV 3.x and Tensorflow on python. This implementation can be extended for the 
evaluation of other models/papers just by following the presented modular design (model/pipeline/preprocess)and accordingly
adjusting **review.py** to dynamically load the appropiate *python module*.
 
In this initial version, each paper/architecture evaluated include: 

1. **Pre-processing** techniques like adaptive histogram equalization, intensity re-scaling, contrast normalization, 
histogram equalization, etc., based on *open-cv* and *sk-image* default implementations.
2. **Model** creation using *Tensorflow* primitives.
3. **Training** using hyper-parameters as described on each paper (for ImageNet or GTSRB).

For all models, a live plot is presented with the validation accuracy after each training epoch. 
