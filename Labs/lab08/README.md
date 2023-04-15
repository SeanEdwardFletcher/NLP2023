# Sentiment Analysis on Tweets about Airlines using RoBERTa

In this notebook, we will be performing sentiment analysis on tweets about airlines using the RoBERTa model. The dataset we will be using is the Twitter US Airline Sentiment dataset from Kaggle.

The dataset contains 14,772 tweets. Each tweet is labeled as positive, negative, or neutral. We will be treating this as a classification problem and building a model that can classify each tweet into one of the three sentiment categories.

## Table of Contents

- [Installation](#Installation)
- [Usage](#Usage)
- [Data-Preprocessing](#Data-Preprocessing)
- [Model-Architecture](#Model-Architecture)
- [Results](#Results)
- [Conclusion](#Conclusion)

## Installation

To run this notebook, you will need to install the following libraries:

    torch
    transformers
    numpy
    pandas

You can install them using pip:

    pip install torch transformers numpy pandas

Additional installs:

    pip install datasets transformers huggingface_hub
    apt-get install git-lfs
    
## Usage

- Download the Twitter US Airline Sentiment dataset from Kaggle.
- Run the notebook cells to preprocess the data, train the model, and evaluate its performance.
- Use the trained model to classify new tweets into sentiment categories.

## Data Preprocessing

First, we will load the dataset and perform some basic preprocessing steps. We will convert all the text to lowercase, remove any URLs, Twitter handles, and any non-alphabetic characters.

## Model Architecture

We will be using the pre-trained RoBERTa model to classify the tweets into sentiment categories. The input to the network will be a sequence of tokenized words. Here are some of the fine-tuned parameters 

    Learning Rate : 1.53e-5
    Batch Size : 16
    Epochs : 2

## Results

After fine-tuning the model, we will evaluate its performance on the test set on its prediction accuracy.

    Accuracy : 89.7

## Conclusion

In this notebook, we have performed sentiment analysis on tweets about airlines using the pre-trained RoBERTa model. We have achieved good accuracy on the test set and demonstrated the effectiveness of RoBERTa in capturing the temporal dependencies between words in a sequence. This model can be used to classify tweets into sentiment categories and help airlines understand customer feedback.
