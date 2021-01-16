# Deep_Learning
## Description 
With all of the tweets circulating every second it is hard to tell whether the sentiment behind a specific tweet will impact a company, or a person's, brand for being viral (positive), or devastate profit because it strikes a negative tone. Capturing sentiment in language is important in these times where decisions and reactions are created and updated in seconds. But, which words actually lead to the sentiment description? In this competition you will need to pick out the part of the tweet (word or phrase) that reflects the sentiment.
The project consists in developing a model for the extraction of feelings, competition rules and datasets used in the following link: https://www.kaggle.com/c/tweet-sentiment-extraction
## Data 
The uploaded data contains csv files of train dataset, test dataset and the submission file
https://www.kaggle.com/c/tweet-sentiment-extraction/data
## Generate Model
### Requirements
For installing requirements : pip install requirements.txt
### GRU model
- The file used for train and generate model is GRU_model.ipynb
- Generated model is Gru_model.hdf5
### LSTM model
- The file used for train and generate model is LSTM_model.ipynb
- Generated model is Lstm_model.hdf5
### Roberta model
- This model is based on RoBERTa pretrained model: A Robustly Optimized BERT Pretraining Approach by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. It is based on Google’s BERT model released in 2018.
https://huggingface.co/transformers/model_doc/roberta.html
- Using Stratified K-Folds cross-validator.It Provides train/test indices to split data in train/test sets. This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
- The file used for train, generate and test model is Roberta_model.py
- Roberta_model.py generate 5 models with jaccard similarity nearly 50 %
- the final file producted for submission is the file https://github.com/achrafBenHamou/Deep_Learning/blob/main/Data/submission.csv

