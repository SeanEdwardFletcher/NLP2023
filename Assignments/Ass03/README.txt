creating_data_sets is a series of functions that read process and write
TSV files for the functions in Ass_03 to use.

plot_the_training_epochs creates a graph out of data from a TSV file

The main file to look at the Ass_03.py.

80-99 gets all the questions and answers from the Posts_law.xml file,
preprocesses the text of the questions and answers and then trains and 
saves a FastText model.
Uncomment to run it.

104 loads an already trained model

177-181 finds and print the P@1 values

434-442 creates data sets

444-451 trains a feed forward neural network model

453-458 saves the data from the training of the model

460-465 tests the modelk
