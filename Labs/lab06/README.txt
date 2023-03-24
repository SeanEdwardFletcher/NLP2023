For running the lab, please note line 202. If you want to train the model on data, comment 
this line out and at the file path to the function call on line 313. If however you want to 
load a previously trained model then add the file path to line 202 and comment out the function
calls on line 313.

Function calls begin at line 309

For plotting the data gathered after each epoch of training, the file plot_the_curves.py reads
a TSV file created in the "with open" block found at line 319 of the lab file.
