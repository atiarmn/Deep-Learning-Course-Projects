Hi!

To execute the code, ensure that the “KidCreative.csv” file is located in the “Source Code” folder (it's already placed there). Then, run the command "python main.py" in the terminal.

The default activation function is the logistic sigmoid, initialized as follows:
L3 = LogisticSigmoidLayer.LogisticSigmoidLayer()

You have the option to switch to other activation functions, all of which are imported in the main.py file. For instance, to use the ReLU activation function, you can modify the code like this:
L3 = ReLULayer.ReLULayer()

Also the default Loss function is the Log Loss, initialized as follows:
L4 = LogLoss.LogLoss()

You have the option to switch to other loss functions, all of which are imported in the main.py file. For instance, to use the Squared error function, you can modify the code like this:
L4 = SquaredError.SquaredError()

Furthermore, in order to utilize the Cross Entropy function, it is necessary to convert the Y array into a one-hot encoding format. I have provided a comment in the code to indicate this step. Please remove the comment to enable this:
n_values = Yhat.shape[1]
Y = np.eye(n_values)[Y]

Also, for multiple classes, you should change the Output size in the FullyConnectedLayer initialization. It is initialized as follows:
L2 = FullyConnectedLayer.FullyConnectedLayer(X.shape[1],1)

You can modify it like this:
L2 = FullyConnectedLayer.FullyConnectedLayer(X.shape[1],4)

Thank you!