# Prototyping-Encoder-Decoder-with-Triplet-Loss
This repository is the official implementation of Prototyping Encoder-Decoder with Triplet Loss: Multimodal Few-Shot Learning for Gait Recognition
# Requirements
To install requirements:
```
pip install -r requirements.txt
```
# Training
To train the models in the paper, run this command:
```
cd code
python training.py
```
We repeat our experiment 20 times. Each time, we select 80% of 30 participants randomly. For each participant in selected 24 people, 75% of unit steps are allocated to the training set and the rest is assigned to the known test data set. In addition, for the remaning 20% of 30 participants(6 people), all unit steps belong to unknown test data set. <br/><br/>
For each repetition, our proposed method is trained and tested independently, then the averaged evaluation metrics are summarized.

# Evaluation
We divide our test set into known and unknown data sets. We define a unit step in the known test data set as true positive (TP) if it is recognized correctly, and false negative (FN) otherwise. Also, we define a unit step in the unknown data set as true negative (TN) if it is not recognized as any known participant, and false positive (FP) otherwise. 
<br/>
We report true postive rate ![equation one](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Csmall%20TPR%3D%20%5Cfrac%7BTP%7D%7BTP%20&plus;%20FN%7D), true negative rate ![equation two](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Csmall%20TNR%3D%20%5Cfrac%7BTN%7D%7BTN%20&plus;%20FP%7D), and accuracy ![equation three](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Csmall%20ACC%20%3D%20%5Cfrac%7BTP%20&plus;%20TN%7D%7BTP%20&plus;%20FN%20&plus;%20TN%20&plus;%20FP%7D)



# Pre-trained Model
You can download pretrained model here: [Pretrained model](https://drive.google.com/file/d/1Rrrc_qbS7MHi9rIxmBGU38nNxuq6AM_y/view?usp=sharing)


# Results
Our model performance:
| Noise rate |     0%    |   5%   |   10%   | 
|------------|-----------|--------|---------|
|Performance |  TPR TNR ACC   | TPR TNR ACC  | TPR TNR ACC|
|lambda = 0  |0.973 0.979 0.976|0.967 0.981 0.974 |0.946 0.983 0.965            |
|lambda = 0.1|0.975 0.994 **0.984**|0.970 0.995 **0.982**|0.954 0.997 **0.976**|
|lambda = 1.0|0.966 0.949 0.958|0.958 0.952 0.955|0.918 0.958 0.938            |

Performance as function of lambda:
![Lambda](images/lambda.png)
Distance Distribution:
![Histogram](images/hist.png)
t-SNE:
![Histogram](images/t-sne.png)

# Contributors
Nelson Minaya nelson.minaya@student.csulb.edu <br/>
Nhat Le nhat.le01@student.csulb.edu
