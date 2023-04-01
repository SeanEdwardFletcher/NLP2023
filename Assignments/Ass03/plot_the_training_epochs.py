
import csv
import matplotlib.pyplot as plt
import numpy as np


dict_of_data = {}

tsv_file = "data_from_each_epoch_simple_2000_lr001_m1_batch1000.tsv"

with open(tsv_file, 'r') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    counter = 1
    for row in reader:
        dict_of_data[str(counter)] = row
        counter += 1


# Define the axis data
x = np.arange(1, 2001)
y = np.arange(0, 20)


# Define the y-axis data for four sets of data
y1 = dict_of_data["4"]
y2 = dict_of_data["3"]
y3 = dict_of_data["2"]
y4 = dict_of_data["1"]

fig = plt.figure(figsize=(8, 6))


# Plot the four sets of data as lines on the same graph
plt.plot(x, y1, label='training losses')
plt.plot(x, y2, label='training accuracies')
plt.plot(x, y3, label='validation losses')
plt.plot(x, y4, label='validation accuracies')

# Add a title and labels for the x and y axes
plt.title('Training/Validation Losses/Accuracies')
plt.xlabel('Number of Epochs')
plt.ylabel('Y')

# Add a legend
plt.legend()

# Display the plot
plt.show()


