# MixMatch
A semi-supervised learning algorithm

# MixMatch

Detailed steps of this semi-supervised learning algorithm



## Data Preparation

- Categorized as labeled data & unlabeled data
- Data preprocessing
  - Drop features where over 95% of the data are missing
  - Impute missing values: -1 for int (discrete values), mean for float (continuous values)
  - Construct matrices after dimension seperation
  - The final data should be in a dimension of 4: [sample size, RGB parameter (3 if colored, 1 if grey scale or non-image data), matrix length, matrix width]

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

    <img src="/Users/hongling/Library/Application Support/typora-user-images/Screen Shot 2020-12-13 at 4.49.19 PM.png" alt="Screen Shot 2020-12-13 at 4.49.19 PM" style="zoom:50%;" />

## Sharpening Pseudo Labels

- Minimize entropy and transform the predictions to a one-hot distribution. Pick the one with noticeably largest value

## Shuffle

- Put together X and U as W; then shuffle W

  - X：augmented labeled dataset and their labels

  - U：augmented unlabeled dataset and their pseudo labels

  - hyper-parameter α = 0.75

    λ ~ Beta(α, α)

    λ' = max(λ, 1-λ)

    x' = λ'x1 + (1-λ')x2

    p' = λ'p1 + (1-λ')p2

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
