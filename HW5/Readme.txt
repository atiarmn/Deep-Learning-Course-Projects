Hi!

To execute the code, ensure that the “yalefaces” folder is located in the “Source Code” folder (It is already placed there). 
Then, to run the CNN for Classification of Synthetic Data codes, run the command "python main.py" in the terminal. To run the CNN For Image Classification codes, run the command "python classification.py" in the terminal.

In these codes, you can see the final architectures. You can change the learning rate with changing the learning_rate variable. Also, you can change the converge limit in the following line with changing `1e-10`.
if(epoch > 0 and (abs(log['train_loss'][-2] - log['train_loss'][-1]) < 1e-10))

Thank you!