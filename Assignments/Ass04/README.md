# Predicting similar questions using base-line sentnce BERT and a fine-tuned sentence BERT with significance testing

In this notebook, we will be performing sentiment analysis on questions classified as similar using sentence BERT models. The dataset we will be using is the contents of the Law Stack Exchange in form of an xml file. After obtaining this file, please label it "Posts_law.xml"


## Table of Contents

- [Installation](#Installation)
- [Usage](#Usage)
- [Data-Preprocessing](#Data-Preprocessing)
- [Results](#Results)


## Installation

To run this notebook, you will need to install the following libraries:
  csv
  numpy
  BeautifulSoup
  sklearn.metrics.pairwise
  re
  transformers
  sentence_transformers
  torch.utils.data
  scipy

    
## Usage

- Download the .xml file from Law Stack Exchange; please label it "Posts_law.xml"
- Run python scripts step_one.py, step_two.py, and step_three.py in order.


## Data Preprocessing

All the question titles and bodies have their html elements removed using beautiful soup


## Results

Base-line Model's Average Mean Reciprocal Rank: 0.002
Fine-tuned Model's Average Mean Reciprocal Rank: 0.005

p-value: 0.014 (the difference between the models' performance is statistically significant)





