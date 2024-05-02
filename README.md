# Deep Residual Learning for Image Recognition
This is my implementation of the Residual Network architecture in PyTorch as desccribed in the 2015 paper *Deep Residual Learning for Image Recognition*.[^1]

It replicates a portion of their CIFAR-10 experiment, as described in section 4.2. It achieves test errors within 0.6% of the paper's results.

## Setup
```
python -m pip install -r requirements.txt
```

## Training
```
python train.py
```

[^1]: He, Kaiming, et al. Deep Residual Learning for Image Recognition. arXiv:1512.03385, arXiv, 10 Dec. 2015. arXiv.org, https://doi.org/10.48550/arXiv.1512.03385.
