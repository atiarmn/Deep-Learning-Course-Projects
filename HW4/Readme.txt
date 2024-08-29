Hi!

To execute the code, ensure that the “MNIST.csv” file is located in the “Source Code” folder (It is already placed there). Then, to run the Multi-class Classification codes, run the command "python main.py" in the terminal. In this code, you can see the final architecture which involves early stopping. You can change the learning rate with changing the learning_rate variable, the ADAM parameters with changing the value of the key 'p1' and 'p2' in adam_args dictionary. Also, you can change the converge limit in the following line with changing `math.pow(10,-6)`.
if(epoch > 0 and (abs(log['train_loss'][-2] - log['train_loss'][-1]) < math.pow(10,-6)))

Moreover, you can change the number of epochs to wait for an improvement in validation loss before stopping the training by changing the `patience` variable.

Also, for the visualization sections, you can run "python VisGrad.py". I commented the codes for each section. 

Thank you!