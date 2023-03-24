# NLP 2023 Lab06
# Sean Fletcher, Maxwell Tuttle

# imports
import csv
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
# end of imports


# function and class definitions
def get_sentence_embedding(model, text):
    # This method takes in the trained model and the input sentence
    # and returns the embedding of the sentence as the average embedding
    # of its words
    words = text.split(" ")
    count = 0
    for i in range(1, len(words)):
        try:
            if count == 0:
                vector = model.wv[words[i]]
            else:
                vector = np.copy(vector + model.wv[words[i]])
            count += 1
        except:
            continue
    return vector / count


def read_tweets_get_vectors(tweet_file_path):
    # This method takes in the file path for the twitter file, and return a
    # dicationary of dictionaries. In the first dictionary the keys are the
    # tweet labels (3 classes), and the values are another dictionary with
    # tweet id as the key and values are tuple of (vector, tweet text)
    df = pd.read_csv(tweet_file_path, sep=',', header=0)
    dic_result = {}
    df1 = df[['tweet_id', 'text', 'airline_sentiment']]
    for index in range(len(df1)):
        try:
            vetor_rep = get_sentence_embedding(word_2_vec_model, df.loc[index, "text"].lower())
            label = df.loc[index, "airline_sentiment"]
            tweet_id = df.loc[index, "tweet_id"]
            if label in dic_result:
                dic_result[label][tweet_id] = (vetor_rep, df.loc[index, "text"].lower())
            else:
                dic_result[label] = {tweet_id: (vetor_rep, df.loc[index, "text"].lower())}
        except:
            pass
    return dic_result


def split_data(twitter_data):
    # takes in the dictionary from the previous step and generate
    # the training, validation, and test sets. Note that the labels
    # are represented as one-hot codings.
    training_x = []
    training_y = []

    validation_x = []
    validation_y = []

    test_x = []
    test_y = []

    for label in twitter_data:

        # labels are indicated as one hot coding [negative, neutral, positive]
        if label == "negative":
            n_label = [1, 0, 0]
        elif label == "neutral":
            n_label = [0, 1, 0]
        else:
            n_label = [0, 0, 1]
        temp_dic = twitter_data[label]
        lst_tweet_ids = list(temp_dic.keys())
        #### Splitting by 80-10-10
        ## Note that you could alternatively use sklearn split method
        train_length = int(len(lst_tweet_ids) * 0.8)
        train_ids = lst_tweet_ids[:train_length]
        remaining = lst_tweet_ids[train_length:]
        test_lenght = int(len(remaining) * 0.5)
        test_ids = remaining[:test_lenght]
        validation_id = remaining[test_lenght:]

        for tweet_id in train_ids:
            training_x.append(temp_dic[tweet_id][0])
            training_y.append(n_label)
        for tweet_id in validation_id:
            validation_x.append(temp_dic[tweet_id][0])
            validation_y.append(n_label)
        for tweet_id in test_ids:
            test_x.append(temp_dic[tweet_id][0])
            test_y.append(n_label)

    # The reason we apply this shuffling is to make sure
    # when passing batches to the network, we see different items
    c = list(zip(training_x, training_y))
    random.shuffle(c)
    training_x, training_y = zip(*c)

    training_x = list(training_x)
    training_x = torch.tensor(training_x)

    training_y = list(training_y)
    training_y = torch.tensor(training_y)

    validation_x = torch.tensor(validation_x)
    validation_y = torch.tensor(validation_y)

    test_x = torch.tensor(test_x)
    test_y = torch.tensor(test_y)

    return training_x, training_y, validation_x, validation_y, test_x, test_y


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, hidden_dim_4, hidden_dim_5, do_01, do_02,
                 output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_1)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim_2, hidden_dim_2)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(hidden_dim_3, hidden_dim_3)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(hidden_dim_3, hidden_dim_4)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(hidden_dim_4, hidden_dim_4)
        self.relu8 = nn.ReLU()
        self.fc9 = nn.Linear(hidden_dim_4, hidden_dim_5)
        self.relu9 = nn.ReLU()
        self.fc10 = nn.Linear(hidden_dim_5, hidden_dim_5)
        self.relu10 = nn.ReLU()
        self.fc11 = nn.Linear(hidden_dim_5, output_dim)
        self.dropout01 = nn.Dropout(do_01)
        self.dropout02 = nn.Dropout(do_02)

        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout01(out)  # do
        # out = self.fc2(out)
        # out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout02(out)  # do
        # out = self.fc4(out)
        # out = self.relu4(out)
        out = self.fc5(out)
        out = self.relu5(out)
        out = self.dropout02(out)  # do
        # out = self.fc6(out)
        # out = self.relu6(out)
        out = self.fc7(out)
        out = self.relu7(out)
        out = self.dropout02(out)  # do
        # out = self.fc8(out)
        # out = self.relu8(out)
        out = self.fc9(out)
        out = self.relu9(out)
        out = self.dropout02(out)  # do
        # out = self.fc10(out)
        # out = self.relu10(out)
        out = self.fc11(out)

        return self.softmax(out)


class ModelModule:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # here goes your parameters
    input_dim = 100
    hidden_dim_1 = 96
    hidden_dim_2 = 48
    hidden_dim_3 = 24
    hidden_dim_4 = 12
    hidden_dim_5 = 6
    do_01 = .66
    do_02 = .33
    output_dim = 3

    ffnn_model = FeedforwardNeuralNetModel(input_dim, hidden_dim_1, hidden_dim_2,
                                           hidden_dim_3, hidden_dim_4, hidden_dim_5,
                                           do_01, do_02, output_dim)
    # ffnn_model.load_state_dict(torch.load('best_model_state.bin'))
    ffnn_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ffnn_model.parameters(), lr=0.01, momentum=0.9)

    def calculate_accuracy(self, y_true, y_pred):
        # this method will be used to calculate the accuracy of your model
        correct = (y_true.argmax(dim=1) == y_pred.argmax(dim=1)).float()
        acc = correct.sum() / len(correct)
        return acc

    def training(self, X_train, Y_train, X_val, Y_val, num_epochs):
        # this method will be used for training your model
        # inputs are the training and validation sets

        # You can define batch size of your choice
        batch_size = 20
        X_train_mini_batches = torch.split(X_train, batch_size)
        Y_train_mini_batches = torch.split(Y_train, batch_size)

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        best_accuracy = 0

        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            epoch_accuracy = 0
            for X_train_mini_batch, Y_train_mini_batch in zip(X_train_mini_batches, Y_train_mini_batches):
                X_train_mini_batch = X_train_mini_batch.to(self.device)
                Y_train_mini_batch = Y_train_mini_batch.to(self.device)
                # forward pass
                train_prediction = self.ffnn_model.forward(X_train_mini_batch.float())
                train_prediction = torch.squeeze(train_prediction)

                # calculating loss
                train_loss = self.criterion(train_prediction.float(), Y_train_mini_batch.float())

                # gradient cleaning
                self.optimizer.zero_grad()

                # getting the gradients
                train_loss.backward()

                # updating parameters
                self.optimizer.step()

                # add each mini batch's loss
                epoch_loss += train_loss.item()

                # add each mini batch's accuracy
                epoch_accuracy += self.calculate_accuracy(Y_train_mini_batch, train_prediction)

            X_val = X_val.to(self.device)
            Y_val = Y_val.to(self.device)

            # forward pass of the validation set
            val_prediction = self.ffnn_model.forward(X_val.float())
            val_prediction = torch.squeeze(val_prediction)

            # calculate loss
            val_loss = self.criterion(val_prediction.float(), Y_val.float())

            # add each mini batch's loss
            validation_loss = val_loss.item()

            # add each mini batch's accuracy
            val_accuracy = self.calculate_accuracy(Y_val, val_prediction)
            if val_accuracy > best_accuracy:
                torch.save(self.ffnn_model.state_dict(), "best_model_state_01.bin")
                best_accuracy = val_accuracy

            epoch_loss /= len(X_train_mini_batches)
            epoch_accuracy /= len(X_train_mini_batches)
            val_losses.append(validation_loss)
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            val_accuracies.append(val_accuracy)

        return train_losses, train_accuracies, val_losses, val_accuracies

    def test(self, X_test, Y_test):
        X_test = X_test.to(self.device)
        Y_test = Y_test.to(self.device)

        # the forward pass
        test_prediction = self.ffnn_model.forward(X_test)
        test_prediction = torch.squeeze(test_prediction)

        # calculate accuracy on test set
        test_accuracy = self.calculate_accuracy(Y_test, test_prediction)

        print("Test Accuracy: ", round(test_accuracy.item(), 4), "\n")

        # # show the report
        # test_prediction = test_prediction.to(self.device)
        # test_prediction = test_prediction.ge(.5).view(-1).cpu()
        # Y_test = Y_test.cpu()
        #
        # print(classification_report(Y_test, test_prediction))
# end of function and class definitions


# function calls
corpus = api.load('text8')
word_2_vec_model = Word2Vec(corpus)
MaxsModelModule = ModelModule()

tr_x, tr_y, va_x, va_y, te_x, te_y = split_data(read_tweets_get_vectors("Tweets.csv"))

t_losses, t_accuracies, v_losses, v_accuracies = MaxsModelModule.training(tr_x, tr_y, va_x, va_y, 1200)


# this "with open" block saves the data from the model's training session to a TSV file to use for later
with open("data_from_each_epoch_01.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(t_losses)
    writer.writerow(t_accuracies)
    writer.writerow(v_losses)
    writer.writerow(v_accuracies)

# testing the model
MaxsModelModule.test(te_x, te_y)

# end of function calls
