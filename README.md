# Prototyping-Encoder-Decoder-with-Triplet-Loss
This repository is the official implementation of Prototyping Encoder-Decoder with Triplet Loss: Multimodal Few-Shot Learning for Gait Recognition
# Requirements
To install requirements:
```
pip install -r requirements.txt
```
# Training
To train the models() in the paper, run this command:
```
cd code/encoders_75
python Batch.py
```

# Evaluation
We divide our testset into known and unknown data sets. We define a unit step in the known test data set as true positive(TP) if it is recognized correctly, and false negative(FN) otherwise. Also, we define a unit step in the unknown data set as true negative(TN) if it is not recognized as any known participant, and false positive(FP) otherwise. 




# Pre-trained Models
You can download pretrained models here: [Pretrained model](https://www.google.com)


# Results
Our model performance:
| Noise rate |     0%    |   5%   |   10%   | 
|------------|-----------|--------|---------|
|Performance | TPR TNR ACC | TPR TNR ACC| TPR TNR ACC|
|lambda = 0  |            |              |           |
|lambda = 0.2|            |              |           |
|lambda = 1.0|            |             |            |

# Contributing
