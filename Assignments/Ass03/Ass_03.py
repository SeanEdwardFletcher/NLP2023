from gensim.models import FastText
import numpy as np
import csv
from scipy import spatial
from post_parser_record import PostParserRecord
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
from creating_data_sets import parse_the_data_tsv_files
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report


def html_tag_remover(input_string):
    """
    this function uses BeautifulSoup to remove the HTML element tags from a string
    :param input_string:
    :return: text: a string without the HTML element tags
    """
    soup = BeautifulSoup(input_string, "html.parser")
    text = soup.get_text()
    return text


def pre_processing(input_text):
    # this function accepts a text (a string), removes the html tags that may be there
    # sepperates that text into sentences, and returns a list of those sentences
    html_free_text = html_tag_remover(input_text)
    list_of_sentences = nltk.sent_tokenize(html_free_text)
    return list_of_sentences


def get_questions_and_answers(post_parser):
    """
    this function finds all the question (titles and bodies) and all the answers to that question
    it then returns:
        a list of strings (the questions)
        a second list of strings (the answers)
    :param post_parser: this is the object created by the PostParserRecord class from the
        post_parser_record.py file. It is a way of navigating the contents of an .xml file
        downloaded from the StackExchange website
    :return: list_of_questions, list_of_answers (both are lists of strings)
    """
    list_of_questions = []
    list_of_answers = []
    for question_id in post_parser.map_questions:
        question_obj = post_parser.map_questions[question_id]
        question_title = question_obj.title
        question_body = question_obj.body
        full_question = question_title + " " + question_body
        list_of_questions.append(full_question)
        if question_obj.answers:
            for answer in question_obj.answers:
                list_of_answers.append(answer.body)
    return list_of_questions, list_of_answers


def train_and_save_the_model(input_list_of_strings):
    model = FastText(vector_size=300, negative=10, window=6, min_n=2, sg=1)
    model.build_vocab(input_list_of_strings)
    model.train(input_list_of_strings, total_examples=len(input_list_of_strings), epochs=10)
    model.save("SeansFT.model")


def get_sentence_embedding(model, sentence):
    # This method takes in the trained model and the input sentence
    # and returns the embedding of the sentence as the average embedding
    # of its words
    words = sentence.split(" ")
    vector = model.wv[words[0]]
    writeable_vector = np.copy(vector)  # Makes a writeable copy of `vector`
    for i in range(1, len(words)):
        writeable_vector += model.wv[words[i]]
    return writeable_vector / len(words)


parsed_law_posts = PostParserRecord("Posts_law.xml")

# # this section of code gets all the questions and answers
# # from the Posts_law.xml file
# # preprocesses the text of the questions and answers
# # and then trains and saves a FastText model
#
# law_questions, law_answers = get_questions_and_answers(parsed_law_posts)
#
# big_list_of_clean_sentences = []
#
# for q_post in law_questions:
#     clean_q_sentences = pre_processing(q_post)
#     big_list_of_clean_sentences += clean_q_sentences
#
# for a_post in law_answers:
#     clean_a_sentences = pre_processing(a_post)
#     big_list_of_clean_sentences += clean_a_sentences
#
# train_and_save_the_model(big_list_of_clean_sentences)



# # loading an already trained model
# the_model = FastText.load('SeansFT.model')




def get_the_p_at_1_scores(model, tsv_file, post_parser_record, bodied=0):
    # if titles_or_bodies == 0, this will find the P@1 scores for titles only
    # if titles_or_bodies != 0, this will find the P@1 scores for bodies only

    embeddings_dict = {}
    high = []
    p_at_1_list = []
    wrong_ids = []

    for test_question_id in post_parser_record.map_questions:
        question_obj = post_parser_record.map_questions[test_question_id]
        if bodied == 0:
            question_text = question_obj.title
        else:
            question_text = question_obj.body
        clean_question_text = html_tag_remover(question_text)
        clean_question_text_embedding = get_sentence_embedding(model, clean_question_text)
        embeddings_dict[test_question_id] = clean_question_text_embedding

    print("The number of questions being compared is (the size of the embeddings_dict):")
    print(len(embeddings_dict.keys()))

    # Open the TSV file for reading
    with open(tsv_file, 'r') as tsvfile:
        # Create a reader object
        reader = csv.reader(tsvfile, delimiter='\t')
        # Loop over each row in the file
        for row in reader:  # each row has a list of two or three question ID numbers

            test_question_id = int(row[0])
            similar_question_ids = [int(row[1])]
            try:
                similar_id_02 = row[2]
                similar_question_ids.append(similar_id_02)
            except IndexError:
                pass

            # grabs the question to test
            try:
                test_question = post_parser_record.map_questions[test_question_id]
            except KeyError:
                wrong_ids.append(test_question_id)
                continue
            if bodied == 0:
                test_text = test_question.title
            else:
                test_text = test_question.body
            clean_test_text = html_tag_remover(test_text)
            test_embedding = get_sentence_embedding(model, clean_test_text)

            highest_dif = -2  # every difference should be higher than -2
            highest_dif_question_id = 0
            for key in embeddings_dict.keys():
                if key == test_question_id:
                    continue
                cosine_similarity = 1 - spatial.distance.cosine(test_embedding, embeddings_dict[key])
                if cosine_similarity > highest_dif:
                    highest_dif = cosine_similarity
                    highest_dif_question_id = key
            if highest_dif_question_id in similar_question_ids:
                p_at_1_list.append(1)
                high.append(1)
            else:
                p_at_1_list.append(0)

    return high, p_at_1_list, wrong_ids


# highh, p_list, wrong_id_list = get_the_p_at_1_scores(the_model, "duplicate_questions.tsv", parsed_law_posts,
#                                                            bodied=1)
# print(highh)
# print(p_list[:20])
# print(wrong_id_list)



class My7LayerFeedforwardNN(nn.Module):
    def __init__(self):
        super(My7LayerFeedforwardNN, self).__init__()

        # # using LeakyReLU
        # self.fc1 = nn.Linear(600, 512)
        # self.relu1 = nn.LeakyReLU()
        # self.fc2 = nn.Linear(512, 256)
        # self.relu2 = nn.LeakyReLU()
        # self.fc3 = nn.Linear(256, 256)
        # self.relu3 = nn.LeakyReLU()
        # self.fc4 = nn.Linear(256, 128)
        # self.relu4 = nn.LeakyReLU()
        # self.fc5 = nn.Linear(128, 64)
        # self.relu5 = nn.LeakyReLU()
        # self.fc6 = nn.Linear(64, 64)
        # self.relu6 = nn.LeakyReLU()
        # self.fc7 = nn.Linear(64, 1)
        # self.dropout00 = nn.Dropout(.33)
        # self.dropout01 = nn.Dropout(.66)

        # # using ReLU complex
        # self.fc1 = nn.Linear(600, 512)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(512, 256)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(256, 256)
        # self.relu3 = nn.ReLU()
        # self.fc4 = nn.Linear(256, 128)
        # self.relu4 = nn.ReLU()
        # self.fc5 = nn.Linear(128, 64)
        # self.relu5 = nn.ReLU()
        # self.fc6 = nn.Linear(64, 64)
        # self.relu6 = nn.ReLU()
        # self.fc7 = nn.Linear(64, 1)
        # self.dropout00 = nn.Dropout(.33)
        # self.dropout01 = nn.Dropout(.66)

        # using ReLU simple
        self.fc1 = nn.Linear(600, 256)
        self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(512, 256)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(256, 256)
        # self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(256, 64)
        self.relu4 = nn.ReLU()
        # self.fc5 = nn.Linear(128, 64)
        # self.relu5 = nn.ReLU()
        # self.fc6 = nn.Linear(64, 64)
        # self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(64, 1)
        self.dropout00 = nn.Dropout(.33)
        self.dropout01 = nn.Dropout(.66)

    def forward(self, x):
        # # complex
        # out = self.fc1(x)
        # out = self.relu1(out)
        # out = self.dropout01(out)  # do 1
        # out = self.fc2(out)
        # out = self.relu2(out)
        # out = self.dropout00(out)  # do 0
        # out = self.fc3(out)
        # out = self.relu3(out)
        # out = self.dropout00(out)  # do 0
        # out = self.fc4(out)
        # out = self.relu4(out)
        # out = self.dropout00(out)  # do 0
        # out = self.fc5(out)
        # out = self.relu5(out)
        # out = self.dropout00(out)  # do 0
        # out = self.fc6(out)
        # out = self.relu6(out)
        # out = self.dropout00(out)  # do 0
        # out = self.fc7(out)

        # simple
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout01(out)  # do 1
        # out = self.fc2(out)
        # out = self.relu2(out)
        # out = self.dropout00(out)  # do 0
        # out = self.fc3(out)
        # out = self.relu3(out)
        # out = self.dropout00(out)  # do 0
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.dropout00(out)  # do 0
        # out = self.fc5(out)
        # out = self.relu5(out)
        # out = self.dropout00(out)  # do 0
        # out = self.fc6(out)
        # out = self.relu6(out)
        # out = self.dropout00(out)  # do 0
        out = self.fc7(out)

        return torch.sigmoid(out)


class ModelModule:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = My7LayerFeedforwardNN()
    # model.load_state_dict(torch.load('best_model_state_00.bin'))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)

    def calculate_accuracy(self, y_true, y_pred):
        # this method will be used to calculate the accuracy of your model
        y_pred = torch.round(y_pred)
        correct = (y_true == y_pred).float()
        acc = correct.sum() / len(correct)
        return acc

    def training(self, X_train, Y_train, X_val, Y_val, num_epochs):
        # inputs are the training and validation sets

        batch_size = 1000
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
            validation_loss = 0
            val_accuracy = 0
            for X_train_mini_batch, Y_train_mini_batch in zip(X_train_mini_batches, Y_train_mini_batches):
                X_train_mini_batch = X_train_mini_batch.to(self.device)
                Y_train_mini_batch = Y_train_mini_batch.to(self.device)

                # forward pass
                train_prediction = self.model.forward(X_train_mini_batch.float())
                train_prediction = torch.squeeze(train_prediction)

                # calculating loss
                train_loss = self.criterion(train_prediction, Y_train_mini_batch)

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
            val_prediction= self.model.forward(X_val.float())
            val_prediction = torch.squeeze(val_prediction)

            # calculate loss
            val_loss = self.criterion(val_prediction, Y_val)

            # add each mini batch's loss
            validation_loss = val_loss.item()

            # add each mini batch's accuracy
            val_accuracy = self.calculate_accuracy(Y_val, val_prediction)
            if val_accuracy > best_accuracy:
                torch.save(self.model.state_dict(), "best_model_state_simple_2000_lr001_m1_batch1000.bin")
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
        test_prediction = self.model.forward(X_test)
        test_prediction = torch.squeeze(test_prediction)

        # calculate accuracy on test set
        test_accuracy = self.calculate_accuracy(Y_test, test_prediction)

        print("Test Accuracy: ", round(test_accuracy.item(), 4), "\n")

        # show the report
        test_prediction = test_prediction.to(self.device)
        test_prediction = test_prediction.ge(.5).view(-1).cpu()
        Y_test = Y_test.cpu()

        print(classification_report(Y_test, test_prediction))


# load the fast text model
the_fasttext_model = FastText.load('SeansFT.model')


def model_ready_data(tsv_file_name, post_parser_record, fast_text_model):

    list_of_tensors = []
    list_of_labels = []

    training_data_set = parse_the_data_tsv_files(tsv_file_name, post_parser_record, fast_text_model)

    for row in training_data_set:
        np_array_01 = row[0]
        t01 = torch.from_numpy(np_array_01)
        np_array_02 = row[1]
        t02 = torch.from_numpy(np_array_02)
        tensor_list = [t01, t02]
        data = torch.cat(tensor_list)

        list_of_tensors.append(data)

        # label_array = np.array([row[2]])
        # t03 = torch.from_numpy(label_array)
        # list_of_labels.append(t03)

        list_of_labels.append(row[2])

    label_array = np.array(list_of_labels)
    t03 = torch.from_numpy(label_array)
    t03 = t03.float()

    tensor_of_tensors = torch.stack(list_of_tensors, 0)
    # tensor_of_labels = torch.stack(list_of_labels, 0)

    return tensor_of_tensors, t03


x_training_data, y_training_data = model_ready_data(
    "feed_forward_training_set_ids.tsv",
    parsed_law_posts,
    the_fasttext_model)

x_validation_data, y_validation_data = model_ready_data(
    "feed_forward_validation_set_ids.tsv",
    parsed_law_posts,
    the_fasttext_model)

Seans_Model_Module = ModelModule()

t_losses, t_accuracies, v_losses, v_accuracies = Seans_Model_Module.training(
    x_training_data,
    y_training_data,
    x_validation_data,
    y_validation_data,
    2000)

with open("data_from_each_epoch_simple_2000_lr001_m1_batch1000.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(t_losses)
    writer.writerow(t_accuracies)
    writer.writerow(v_losses)
    writer.writerow(v_accuracies)

x_test_data, y_test_data = model_ready_data(
    "feed_forward_test_set_ids.tsv",
    parsed_law_posts,
    the_fasttext_model)

Seans_Model_Module.test(x_test_data, y_test_data)