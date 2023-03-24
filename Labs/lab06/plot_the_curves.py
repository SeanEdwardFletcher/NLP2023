# NLP 2023 Lab06
# Sean Fletcher, Maxwell Tuttle

import csv
import matplotlib.pyplot as plt
import numpy as np


dict_of_data = {}

with open("data_from_each_epoch_00.tsv", 'r') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    counter = 1
    for row in reader:
        dict_of_data[str(counter)] = row
        counter += 1


# Define the x-axis data
x = np.arange(1, 1201)
y = np.arange(1, 100)


# Define the y-axis data for four sets of data
y1 = dict_of_data["1"]
y2 = dict_of_data["2"]
y3 = dict_of_data["3"]
y4 = dict_of_data["4"]

# Plot the four sets of data as lines on the same graph
plt.plot(x, y1, label='training losses')
plt.plot(x, y2, label='training accuracies')
plt.plot(x, y3, label='validation losses')
plt.plot(x, y4, label='validation accuracies')

# Add a title and labels for the x and y axes
plt.title('Training/Validation Losses/Accuracies')
plt.xlabel('Number of Epochs')

# Add a legend
plt.legend()

# Display the plot
plt.show()
