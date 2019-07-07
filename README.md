# Model-Pruning-on-Audio-Event-Detection-Task
explore model pruning methods on Audio Event Detection Task

The dataset and task is based on the same problem in my another project [Audio-Classification-in-Multi-stage-Semi-Supervised-Learning-Way
](https://github.com/JiachuanDENG/Audio-Classification-in-Multi-stage-Semi-Supervised-Learning-Way) But we only focus on exploring model pruning in this project.

## Problem Definition: 
Given a wav file (in variable length), predict its corresponding label(s), each wav could be in multiple classes
![task2_freesound_audio_tagging](https://user-images.githubusercontent.com/20760190/59892155-d9c85480-938c-11e9-8e64-65582cec6b32.png)

## Data Set:
Original Dataset can be found in Kaggle: https://www.kaggle.com/c/freesound-audio-tagging-2019/data.

To save time in data preprocessing, we also use the processed dataset (converting raw wav data to numpy matrix with Logmel transformation) https://www.kaggle.com/daisukelab/fat2019_prep_mels1

And since a more complicated but better performance solution has been explored in my project [Audio-Classification-in-Multi-stage-Semi-Supervised-Learning-Way
](https://github.com/JiachuanDENG/Audio-Classification-in-Multi-stage-Semi-Supervised-Learning-Way), in this project, we only focus on exploring model pruning, so we will only use the curated data (i.e. *mels_train_curated.pkl*) for simplicity.

## Method:
In our code, we  implemented a CNN based model.

Since CNN type model only allow fixed length input, while the data input length in our dataset is variable, we need to cut the long input audio into segments with fixed length (padding if necessary), 
and use the average of each segment's prediction as final prediction of the original audio data.


![Screen Shot 2019-06-20 at 7 20 22 PM](https://user-images.githubusercontent.com/20760190/59893091-8ce67d00-9390-11e9-92c4-5529ae6c0ff7.png)


## Model Pruning
The original model we use are quite (not necessarily) large, we explore model pruning (based on paper [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/pdf/1708.06519.pdf)) in this project. 

And accroding to paper [RETHINKING THE VALUE OF NETWORK PRUNING](https://arxiv.org/pdf/1810.05270.pdf), rather than fine tune the pruned model, we retrain the pruned from scratch by default (only consiter the model pruning as a model architecture
searching technique, and can get even better performance than unpruned model).

## Run the code
Make sure data path in config.ini is modified correctly according to the data you downloaded. 

Then run ``` python3 runme.py ```

## Experiment Result

Evaluation Metrics we use is [ ***lwlrap: label-weighted label-ranking average precision***](https://scikit-learn.org/stable/modules/model_evaluation.html#label-ranking-average-precision)

***FLOPS*** and ***Params*** are calculated with package from [THOP: PyTorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)

| Model | Validation (lwlrap) | Testing (lwlrap) | FLOPS (G) | Params (M)| Testing Time (s)
| ------------- | ------------- | ----------- | -----------| -----------| -----------
| Unpruned |0.758 | 0.739 | 3.37 | 4.763| 31.4 
| Pruned | 0.787 | 0.775 | 0.77 | 1.139 | 12.1 

We find by pruning **50%** number of channels in conv layers, we can achieve 4x less FLOPS/ Params, and more than 2x faster inferencing computation.

