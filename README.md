# MixMatch

This repository is about steps of this semi-supervised learning algorithm.
- Original paper: [*MixMatch - A Holistic Approach to Semi-Supervised Learning*](https://arxiv.org/abs/1905.02249) by David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver and Colin Raffel.
- Code provided by Google Research is [here](https://github.com/google-research/mixmatch). 

This algorithm is originally designed for image classification and usually requires CUDA support. Together with Yang Wan and Minchenxi Zhou, I used this approach and modified the code for tabular data and a CPU-only environment in my internship. I cannot upload code here because of confidentiality requirement, but I can share my understanding of this algorithm because it's public content:)



## Data Preparation

- Categorized as labeled data & unlabeled data
- Data preprocessing
  - Drop features where over 95% of the data are missing
  - Impute missing values: -1 for int (discrete values), mean for float (continuous values)
  - Construct matrices after dimension seperation
  - The final data should be in a dimension of 4
    - sample size
    - RGB parameter (3 if colored, 1 if grey scale or non-image data)
    - matrix length
    - matrix width

## Data Augmentation

- Augmentation times: 1 for labeled data; K (hyper parameter) for unlabeled data
- For image data
  - Strong augmentation: sharpening, adjusting saturation, and adjusting color temperature
  - Weak augmentation: translation, rotation, and cropping
- For tabular data
  - Random flipping and cropping of matrices (substituting the margins with all 0)

## Label Guessing

- Generate pseudo labels for unlabeled data with models. Try multiple times and take the mean as the final result
- For image data, Wide-ResNet-28 (28 layers of wide residual networks) is commonly used. However, technically other unsupervised learning models could work too, depending on the data format
  - Wide-Res-Net structure
  <img src="https://github.com/HonglingLei/MixMatch/blob/main/WideResNet_structure.png" />

## Sharpening Pseudo Labels

- Minimize entropy and transform the predictions to a one-hot distribution. Pick the one with noticeably largest value

## Shuffle

- Put together X and U as W; then shuffle W
  - X：augmented labeled dataset and their labels
  - U：augmented unlabeled dataset and their pseudo labels
  - hyper-parameter α = 0.75
    - λ ~ Beta(α, α)
    - λ' = max(λ, 1-λ)
    - x' = λ'x1 + (1-λ')x2
    - p' = λ'p1 + (1-λ')p2

## MixUp

- Mix up every element in W
- Take maximums for λ & 1-λ, so that (x1, p1) still takes up a principal component in the final result after mixup. In this way, X's can still represent labeled data while U's can represent unlabeled ones

## Mixed Loss Calculation

- Calculate cross entropy for labeled data
- Calculate the L2 distance between predictions and pseudo labels for unlabeled data
- L = Lx + λuLu
  - hyper-parameter λu = 100

## Model Training & Parameter Tuning

- Train the model with Wide-Res-Net again
- Tune parameters until the model performs well
