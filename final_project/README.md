# SimpleText Task 1

This repository includes all of the programs and optimal results for my team's (Team Fletcher, one member, me) submission for SimpleText Task 1.

## Table of Contents

- [Installation](#Installation)
- [Text-Files](#Text-Files)
- [Steps-to-Run](#Steps-to-Run)
- [Model-and-Results](#Model-and-Results)
- [Baseline-Results](#Baseline-Results)
- [Our-Results](#Our-Results)
- [Conclusion](#Conclusion)

## Installation

To run this code, you will need access to the [dataset](http://simpletext-project.com/2023/clef/) for task 1 of SimpleText CLEF lab, and login details to access their servers. I will be assuming you have access for the rest of the explanation.

The necessary installs for the code are as such

    torch
    tqdm
    sentence_transformers
    transformers
    textstat
    ranx

You can install them using pip:

    pip install torch tqdm sentence_transformers transformers textstat ranx
    
Clone the repository all to one folder to properly run. Directories may need to be changed to fit your machine.

## Text Files
- elastic_baseline_results.txt -> Unmodified results from elastic search of queries
- donut_graph_run1.txt -> Reranking of baseline by cross encoder
- donut_graph_run1_combined_with_baseline_n.txt -> Normalized combined results 

## Steps to Run

- Get the SimpleText dataset from CLEF
- Download the jsons using download_elastic.py
- Read the jsons to a baseline results txt file using readJSON.py
- Run cross_encoder.py with the baseline results file and new results file as arguments
- Run evaluation.py with the new results file (use the combined and normalized version) as the argument
- Run get_readability_scores_with_json.py with the new results file (use the combined and normalized version) as the argument

## Model and Results

Our final results come from a combination of the ms-marco-electra-base cross encoder and the intial baseline results from elastic search. The cross encoder does its own reranking of the initial results from elastic search, then that output is directed into a comination program that does a final reranking using a combination of their scores. This results in higher NDCG and MAP scores than both individual results.

## Baseline Results

    NDCG@10: 0.399
    MAP: 0.461
    flesch average: 31.58531276140922
    smog average: 14.593094153731943
    Coleman Liau average: 15.39346035876951

## Our Results

    NDCG@10: 0.464
    MAP: 0.505
    flesch average: 31.585312761409227
    smog average: 14.593094153731947
    Coleman Liau average: 15.39346035876951

## Conclusion

With the use of a cross encoder and a combination algorithm, we were able to improve the baseline results. If we were to continue on the project, we would attempt to fine-tune the cross encoder on the labelled data and implement a way to select more readable passages than just the entire abstract (as our current implementation has no improvement in the readability department). 





To run this project:

main.py
All function calls are in main.py
You will need to add three directory paths at the top of main.py
These directories will be where you save your downloaded json files of scientific articles.
You will need to add the file path to your training data.

download_elastic.py
You will need to add a directory path and your user name and password to the file download_elastic.py

evaluation.py
Pass the text files created from some of the dictionaries to this file using your terminal to get your scores
ex. ->  python evaluation.py text_file_to_evaluate


In main.py I do not suggest running the whole file at once. Look at each section and run the sections you need sepperately.


There are sections that focus on downloading relevant json files of scientific abstracts.

There are sections that make calls using python's requests library

There are sections that use YAKE to extract keywords from various sources.

There are sections that save dicionaries as text documents that can be passed to the evauation.py file using the terminal 

There are sections that prepare data, and then fine-tune a sentence bert model.

