import backend
import nn

"""
Question 1 (6 points): Perceptron

Notes:
    Similar to a Keras Sequential model using:

        # New model
        model = tf.keras.models.Sequential()

        # Add 1 layer with 1 node
        model.add(tf.keras.layers.Dense(1, input_dim=INPUT_DIMENSION, activation="relu"))

        # Classification model using Stochastic Gradient Descent as the optimizer
        model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])

        # Train the model
        model.fit(
            x=x,
            y=y,
            batch_size=BATCH_SIZE,  # Should be 1
            epochs=EPOCHS,  # Should be 10000
            # verbose="auto",
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss')],
        )

Result:
    Question q1
    ===========
    *** q1) check_perceptron
    Sanity checking perceptron...
    Sanity checking perceptron weight updates...
    Sanity checking complete. Now training perceptron
    *** PASS: check_perceptron

    ### Question q1: 6/6 ###

    Finished at 20:17:52

    Provisional grades
    ==================
    Question q1: 6/6
    ------------------
    Total: 6/6
"""


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w: nn.Parameter = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"

        # print("run")

        """
        Implement the run(self, x) method. This should compute the dot product of the stored weight vector and the 
        given input, returning an nn.DotProduct object.
        """
        dot_product = nn.DotProduct(self.w, x)

        # print(dot_product)

        return dot_product

    def get_prediction(self, x: nn.Constant):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # print("get_prediction")

        """
        Implement get_prediction(self, x), which should return 1 if the dot product is non-negative or −1 otherwise. 
        You should use nn.as_scalar to convert a scalar Node into a Python floating-point number.
        """
        v = nn.as_scalar(self.run(x))

        return 1 if v >= 0 else -1

    def train(self, dataset: backend.Dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"

        """
        Write the train(self) method. This should repeatedly loop over the data set and make updates on examples 
        that are misclassified. Use the update method of the nn.Parameter class to update the weights. When an 
        entire pass over the data set is completed without making any mistakes, 100% training accuracy has 
        been achieved, and training can terminate.
        """
        # print("#" * 100)
        # print("train")

        # ***** Hyper Parameters *****
        EPOCHS = 100000  # There are always epochs
        BATCH_SIZE = 1  # Stochastic gradiant descent because of batch size == 1

        """
        When an entire pass over the data set is completed without making any mistakes, 
        100% training accuracy has been achieved, and training can terminate.
        
        Notes:
            A simple "early stopping" implementation without the existence of a validation dataset to compare to 
            which is why the variable is "pseudo' because we only have the training dataset.
        """
        early_stopping_pseudo = False

        # This should repeatedly loop over the data set and make updates on examples that are misclassified.
        for _ in range(EPOCHS):

            """
            When an entire pass over the data set is completed without making any mistakes, 
            100% training accuracy has been achieved, and training can terminate.
            """
            if early_stopping_pseudo:
                break

            """
            When an entire pass over the data set is completed without making any mistakes, 
            100% training accuracy has been achieved, and training can terminate.
            """
            early_stopping_pseudo = True

            # Loop over entire dataset
            for x, y in dataset.iterate_once(BATCH_SIZE):

                x_prediction = self.get_prediction(x)
                y_true = nn.as_scalar(y)

                # x_dot_product = nn.as_scalar(self.run(x))

                # Train only when prediction is wrong
                if x_prediction != y_true:
                    # print("{:<20}{:<5}{:<5}".format(x_dot_product, x_prediction, y_true))

                    """
                    When an entire pass over the data set is completed without making any mistakes, 
                    100% training accuracy has been achieved, and training can terminate.
                    """
                    early_stopping_pseudo = False

                    """
                    Use the update method of the nn.Parameter class to update the weights. 
                    
                    Notes:
                        self.w.update(direction, multiplier)    
                            direction is a Node 
                            multiplier is a python scalar
                    """
                    self.w.update(x, y_true)


"""
Question 2 (Not Required): Non-linear Regression

Notes:
    Similar to a Keras Sequential model using:

        # New model
        model = tf.keras.models.Sequential()

        # 1 Input, 3 Hidden, 1 Output
        model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(batch_size * 1,)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        
        # Classification model using Stochastic Gradient Descent as the optimizer
        model.compile(loss='MSE', optimizer='SGD', metrics=['accuracy'])

        # Train the model
        model.fit(
            x=x,
            y=y,
            batch_size=BATCH_SIZE,  # Should be 1
            epochs=EPOCHS,  # Should be 10000
            # verbose="auto",
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss')],
        )
        
Reference:
    What is the advantage of using an InputLayer (or an Input) in a Keras model with Tensorflow tensors?
        Notes:
            Keras notes
            
        Reference:
            https://stackoverflow.com/questions/45217973/what-is-the-advantage-of-using-an-inputlayer-or-an-input-in-a-keras-model-with
    
    What is the point of having a dense layer in a neural network with no activation function?
        Notes:
            Why there is no activation function on the last layer for regression using Keras
            
            "One such scenario is the output layer of a network performing regression, which should be naturally linear. 
            This tutorial demonstrates this case."
            
                https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

        Reference:
            https://stats.stackexchange.com/questions/361066/what-is-the-point-of-having-a-dense-layer-in-a-neural-network-with-no-activation
            
            
Result:
    Question q2
    ===========
    *** q2) check_regression
    Your final loss is: 0.000010
    *** PASS: check_regression
    
    ### Question q2: 6/6 ###
    
    Finished at 3:01:49
    
    Provisional grades
    ==================
    Question q2: 6/6
    ------------------
    Total: 6/6
"""


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        ####################
        """
        Hyper Hyper Parameters
        """
        # Used for scaling the learning rate and batch size appropriately for faster training
        self.multiplier = 50

        ####################
        """
        Hyper Parameters
        
        """
        self.learning_rate = .001
        self.batch_size = 1
        self.EPOCHS = 100000  # There are always epochs (DON'T CHANGE THIS)

        ####################
        """
        Early Stopping
        """

        # When to stop training if loss is val loss is low and loss tracks val loss well
        self.early_stopping_threshold = 0.00001

        ####################
        """
        Stuff the user should not see because it's meta stuff
        
        Notes:
            Basically, we aer multiplying a hyper parameter by a hyper hyper parameter
            
        Reference:
            15. Batch Size and Learning Rate in CNNs
                Notes:
                    " there's actually a subtle point there that it's important to know which is
                    that as you increase the batch size you should also increase the
                    learning rate so if you multiply the batch size by four you should also
                    generally speaking and multiply the learning rate by four so with bigger
                    batches because you're averaging over larger number samples you can get away
                    with higher learning rates than you otherwise could..."
                    
                Reference:
                    https://www.youtube.com/watch?v=ZBVwnoVIvZk
        """
        self.learning_rate = self.learning_rate * self.multiplier
        self.batch_size = self.batch_size * self.multiplier

        ####################
        """
        Reduce learning rate on plateau (reliant on early stopping)
        
        Notes:
            This will basically make training faster because the learning rate will be smaller when getting
            closer to the the self.early_stopping_threshold
            
            Basically Reduce learning rate when the new loss (val loss) is far from the previous loss (val loss) 

            This is similar to Keras's ReduceLROnPlateau callback function, but
            it reduces the learning rate when the ratio between the old loss over the new loss abs value is greater 
            than self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value
        
        IMPORTANT NOTES:
            THIS METHOD IS UNRELIABLE FOR VERY SMALL self.early_stopping_threshold SUCH AS
            A VALUE LESS THAN 0.00001 AND IF YOUR MODEL IS NOT VERY COMPLEX (NOT ENOUGH NEURONS AND/OR LAYERS).
        
        """

        # Multiply this value by self.learning_rate to get a new self.learning_rate
        self.reduce_learning_rate_multiple = 0.50  # Reduce the learning rate by half

        self.reduce_learning_rate_amount_of_reductions = 3

        # Previous loss to compare to
        self._loss_val_previous = None

        # If the ratio of the new loss over the old loss is less than 1 then reduce the learning rate
        self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value = self.early_stopping_threshold * 2

        """
        Because we reduced the learning rate, we must also adjust the threshold that led to that change
        in learning rate because the new learning rate will make us reach the threshold again very quickly
        which will then make the learning rate really small and the model training will take forever... 
        """
        self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value_ratio = self.reduce_learning_rate_multiple * 0.01

        ####################
        """
        Model
        """

        # Input layer
        """
        Parameter (weight 1)'s first argument is the size of the input.

        Parameter (weight 1)'s second argument must be the input for the following parameter (bias 1)'s second argument.
        This value can be anything you choose.
        The statement above is only true for the hidden layers, not the last layer.

        Keras
            tf.keras.layers.Dense(32, activation='relu', input_shape=(batch_size * 1,))
        """
        self.w1 = nn.Parameter(1, 32)
        self.b1 = nn.Parameter(1, 32)

        # Hidden layer 1
        """
        Parameter (weight 2)'s first argument must be the same for the previous parameter (bias 1)'s second argument.
        The statement above is only true for non input layers.
        
        Parameter (weight 2)'s second argument must be the input for the following parameter (bias 2)'s second argument.
        This value can be anything you choose.
        The statement above is only true for the hidden layers, not the last layer.
        
        Keras
            tf.keras.layers.Dense(64, activation='relu')
        """
        self.w2 = nn.Parameter(32, 64)
        self.b2 = nn.Parameter(1, 64)

        # Hidden layer 2
        """
        Parameter (weight 3)'s first argument must be the same for the previous parameter (bias 2)'s second argument.
        The statement above is only true for non input layers.

        Parameter (weight 3)'s second argument must be the input for the following parameter (bias 3)'s second argument.
        This value can be anything you choose.
        The statement above is only true for the hidden layers, not the last layer.
        
        Keras
            tf.keras.layers.Dense(64, activation='relu')
        """
        self.w3 = nn.Parameter(64, 64)
        self.b3 = nn.Parameter(1, 64)

        # Hidden layer 3
        """
        Parameter (weight 4)'s first argument must be the same for the previous parameter (bias 3)'s second argument.
        The statement above is only true for non input layers.

        Parameter (weight 4)'s second argument must be the input for the following parameter (bias 4)'s second argument.
        This value can be anything you choose.
        The statement above is only true for the hidden layers, not the last layer.

        Keras
            tf.keras.layers.Dense(32, activation='relu')
        """
        self.w4 = nn.Parameter(64, 32)
        self.b4 = nn.Parameter(1, 32)

        # Output layer
        """
        Parameter (weight 5)'s first argument must be the same for the previous parameter (bias 4)'s second argument.
        The statement above is only true for non input layers.

        Parameter (weight 5)'s second argument must be the output size which is 1 because this is linear regression.
        For example:
            y = mx + b where m is what you are trying to solve. Notices that it is 1 value which is a scalar which this
            model tries to approximate.
        
        Keras
            tf.keras.layers.Dense(1)
        """
        self.w5 = nn.Parameter(32, 1)
        self.b5 = nn.Parameter(1, 1)

        self.parameters = [self.w1, self.b1,
                           self.w2, self.b2,
                           self.w3, self.b3,
                           self.w4, self.b4,
                           self.w5, self.b5,
                           ]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        """
        Notes:
            From notes
                relu(x * w1 + b1) * W_2 + b_2

            What was created
                relu(relu(relu(x * w1 + b1) * W_2 + b_2) * W_3 + B_3) * w_4 + b_4
        
        """
        # Input layer (Mimics hidden layer because why not)
        x_mul_w1 = nn.Linear(x, self.w1)
        x_mul_w1_add_b1 = nn.AddBias(x_mul_w1, self.b1)
        relu_1 = nn.ReLU(x_mul_w1_add_b1)
        layer_input = relu_1

        # Hidden Layer 1
        relu_1_mul_w2 = nn.Linear(layer_input, self.w2)
        relu_1_mul_w2_add_b2 = nn.AddBias(relu_1_mul_w2, self.b2)
        relu_2 = nn.ReLU(relu_1_mul_w2_add_b2)
        layer_hidden_1 = relu_2

        # Hidden Layer 2
        relu_2_mul_w3 = nn.Linear(layer_hidden_1, self.w3)
        relu_2_mul_w3_add_b3 = nn.AddBias(relu_2_mul_w3, self.b3)
        relu_3 = nn.ReLU(relu_2_mul_w3_add_b3)
        layer_hidden_2 = relu_3

        # Hidden Layer 3
        relu_3_mul_w4 = nn.Linear(layer_hidden_2, self.w4)
        relu_3_mul_w4_add_b4 = nn.AddBias(relu_3_mul_w4, self.b4)
        relu_4 = nn.ReLU(relu_3_mul_w4_add_b4)
        layer_hidden_3 = relu_4

        # Output Layer
        relu_4_mul_w5 = nn.Linear(layer_hidden_3, self.w5)
        relu_4_mul_w5_add_b5 = nn.AddBias(relu_4_mul_w5, self.b5)
        relu_5 = nn.ReLU(relu_4_mul_w5_add_b5)  # CANNOT USE THIS FOR layer_output
        layer_output = relu_4_mul_w5_add_b5

        return layer_output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        forward_pass = self.run(x)

        # We are doing Regression so teh loss function is Mean Squared Error (MSE)
        return nn.SquareLoss(forward_pass, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        for _ in range(self.EPOCHS):

            # print(nn.Constant(dataset.x), nn.Constant(dataset.y))

            for x, y in dataset.iterate_once(self.batch_size):

                loss = self.get_loss(x, y)
                # print(loss)
                # print(nn.as_scalar(loss))

                gradient = nn.gradients(loss, self.parameters)
                # print(gradient)

                # Only works for the first gradient[0] because the shape I think...
                # print(nn.as_scalar(nn.DotProduct(gradient[0], gradient[0])))

                # Update parameters
                for index, parameter in enumerate(self.parameters):
                    parameter.update(gradient[index], self.learning_rate * -1)

            """
            Validation loss (pseudo). It is pseudo because the training dataset is the testing dataset.
            
            Notes:
                How far the model is from the actual true data
            """
            loss_val_pseudo = nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)))
            # print(loss_val_pseudo)

            """
            No validation data is available for this dataset. In this assignment, only the Digit Classification and 
            Language Identification datasets have validation data.
            
            Notes:
                Can't use the print below because it does not exist
            """
            # print(dataset.get_validation_accuracy())

            if loss_val_pseudo < self.early_stopping_threshold:
                break

            """
            Reduce learning rate on plateau
    
            Notes:
                This is similar to Keras's ReduceLROnPlateau callback function, but
                it reduces the learning rate when the ratio between the old loss over the new loss abs value is less 
                than self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value
            """
            loss_val_new = abs(loss_val_pseudo)

            if self._loss_val_previous is not None:

                # Ratio between val loss and early stopping threshold (EST)
                ratio_val_loss_new_over_old_abs_value = abs(1 - (loss_val_new / self._loss_val_previous))

                # print(ratio_val_loss_new_over_old_abs_value)

                """
                If the amount of times already reduced is > 0 and
                If the ratio between the old loss over the new loss abs value is less than 
                self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value then reduce the learning rate 
                """
                if (self.reduce_learning_rate_amount_of_reductions > 0 and (
                        ratio_val_loss_new_over_old_abs_value < self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value)):
                    # print(self.learning_rate, self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value)

                    _learning_rate_old = self.learning_rate

                    # Reduce the learning rate
                    self.learning_rate *= self.reduce_learning_rate_multiple

                    # Reduce the threshold even more
                    self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value *= self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value_ratio

                    # Reduce the amount of time to reduce the learning rate
                    self.reduce_learning_rate_amount_of_reductions -= 1

                    print("Reducing learning rate from {} to {}".format(_learning_rate_old, self.learning_rate))
                    # print(self.learning_rate, self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value)

            self._loss_val_previous = loss_val_new


"""
Question 3 (Not Required): Digit Classification

Notes:

    Similar to a Keras Sequential model using:

        # New model
        model = tf.keras.models.Sequential()

        # 1 Input, 3 Hidden, 1 Output
        model.add(tf.keras.layers.Dense(1024, activation='relu', input_shape=(self.image_flattened_dimensions,)))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(124, activation='relu'))
        model.add(tf.keras.layers.Dense(10), activation='softmax')  # 10 for 10 digits (0 to 9)

        # Classification model using Stochastic Gradient Descent as the optimizer
        model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

        # Train the model
        model.fit(
            x=x,
            y=y,
            batch_size=BATCH_SIZE,  # Should be 1
            epochs=EPOCHS,  # Should be 10000
            # verbose="auto",
            callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='loss'), 
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss'),
            ],
        )

IMPORTANT NOTES:
    THIS MODEL TAKES OVER 2 MINUTES TO RUN

Reference:
    What is the advantage of using an InputLayer (or an Input) in a Keras model with Tensorflow tensors?
        Notes:
            Keras notes

        Reference:
            https://stackoverflow.com/questions/45217973/what-is-the-advantage-of-using-an-inputlayer-or-an-input-in-a-keras-model-with

    What is the point of having a dense layer in a neural network with no activation function?
        Notes:
            Why there is no activation function on the last layer for regression using Keras

            "One such scenario is the output layer of a network performing regression, which should be naturally linear. 
            This tutorial demonstrates this case."

                https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

        Reference:
            https://stats.stackexchange.com/questions/361066/what-is-the-point-of-having-a-dense-layer-in-a-neural-network-with-no-activation


Result:
    Question q3
    ===========
    *** q3) check_digit_classification
    Validation Accuracy: 0.9478
    Validation Accuracy: 0.9608
    Validation Accuracy: 0.969
    Validation Accuracy: 0.9724
    Your final test set accuracy is: 97.400000%
    *** PASS: check_digit_classification
    
    ### Question q3: 6/6 ###
    
    Finished at 1:58:00
    
    Provisional grades
    ==================
    Question q3: 6/6
    ------------------
    Total: 6/6
"""


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        """

        Notes:
            MNIST Example
            This is pretty much a copy of Q2

        """
        ####################
        """
        Special stuff
        """

        """
        This accuracy is used instead of loss and val loss because we are dealing with images
        and it's very difficult to get a good model to identify all images correctly
        
        Notes:
            To get max points for grading:
                Your final test set accuracy (96.840000%) must be at least 97% to receive full points for this question
        """
        self.val_accuracy_sufficient_threshold = .97

        ####################
        """
        Hyper Hyper Parameters
        """
        # Used for scaling the learning rate and batch size appropriately for faster training
        self.multiplier = 50

        ####################
        """
        Hyper Parameters

        """
        self.learning_rate = .001
        self.batch_size = 1
        self.EPOCHS = 100000  # There are always epochs (DON'T CHANGE THIS)

        ####################
        """
        Early Stopping
        """

        # When to stop training if loss is val loss is low and loss tracks val loss well
        self.early_stopping_threshold = 0.00001

        ####################
        """
        Stuff the user should not see because it's meta stuff

        Notes:
            Basically, we aer multiplying a hyper parameter by a hyper hyper parameter

        Reference:
            15. Batch Size and Learning Rate in CNNs
                Notes:
                    " there's actually a subtle point there that it's important to know which is
                    that as you increase the batch size you should also increase the
                    learning rate so if you multiply the batch size by four you should also
                    generally speaking and multiply the learning rate by four so with bigger
                    batches because you're averaging over larger number samples you can get away
                    with higher learning rates than you otherwise could..."

                Reference:
                    https://www.youtube.com/watch?v=ZBVwnoVIvZk
        """
        self.learning_rate = self.learning_rate * self.multiplier
        self.batch_size = self.batch_size * self.multiplier

        ####################
        """
        Reduce learning rate on plateau (reliant on early stopping)

        Notes:
            This will basically make training faster because the learning rate will be smaller when getting
            closer to the the self.early_stopping_threshold

            Basically Reduce learning rate when the new loss (val loss) is far from the previous loss (val loss) 

            This is similar to Keras's ReduceLROnPlateau callback function, but
            it reduces the learning rate when the ratio between the old loss over the new loss abs value is greater 
            than self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value

        IMPORTANT NOTES:
            THIS METHOD IS UNRELIABLE FOR VERY SMALL self.early_stopping_threshold SUCH AS
            A VALUE LESS THAN 0.00001 AND IF YOUR MODEL IS NOT VERY COMPLEX (NOT ENOUGH NEURONS AND/OR LAYERS).

        """

        # Multiply this value by self.learning_rate to get a new self.learning_rate
        self.reduce_learning_rate_multiple = 0.50  # Reduce the learning rate by half

        self.reduce_learning_rate_amount_of_reductions = 3

        # Previous loss to compare to
        self._loss_val_previous = None

        # If the ratio of the new loss over the old loss is less than 1 then reduce the learning rate
        self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value = self.early_stopping_threshold * 2

        """
        Because we reduced the learning rate, we must also adjust the threshold that led to that change
        in learning rate because the new learning rate will make us reach the threshold again very quickly
        which will then make the learning rate really small and the model training will take forever... 
        """
        self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value_ratio = self.reduce_learning_rate_multiple * 0.01

        ####################

        """
        Model
        
        Notes:            
            This NN flattened the entire image 
        
        IMPORTANT NOTES:
            DON'T MAKE THE MODEL VERY BIG OR TRAINING WILL TAKE FOREVER!
        
        Reference:
            But what is a neural network? | Chapter 1, Deep learning
                Notes:
                    Same process
                Reference:
                    https://www.youtube.com/watch?v=aircAruvnKk
        """
        self.image_flattened_dimensions = 28 ** 2  # 784 x 1

        # Input layer
        """
        Parameter (weight 1)'s first argument must be the size of the input.

        Parameter (weight 1)'s second argument must be the input for the following parameter (bias 2)'s second argument.
        This value can be anything you choose.
        The statement above is only true for the hidden layers, not the last layer.

        Keras
            model.add(tf.keras.layers.Dense(1024, activation='relu', input_shape=(self.image_flattened_dimensions,)))
        """
        self.w1 = nn.Parameter(self.image_flattened_dimensions, 1024)
        self.b1 = nn.Parameter(1, 1024)

        # Hidden layer 1
        """
        Parameter (weight 2)'s first argument must be the same for the previous parameter (bias 1)'s second argument.
        The statement above is only true for non input layers.

        Parameter (weight 2)'s second argument must be the input for the following parameter (bias 2)'s second argument.
        This value can be anything you choose.
        The statement above is only true for the hidden layers, not the last layer.

        Keras
            tf.keras.layers.Dense(512, activation='relu')
        """
        self.w2 = nn.Parameter(1024, 512)
        self.b2 = nn.Parameter(1, 512)

        # Hidden layer 2
        """
        Parameter (weight 3)'s first argument must be the same for the previous parameter (bias 2)'s second argument.
        The statement above is only true for non input layers.

        Parameter (weight 3)'s second argument must be the input for the following parameter (bias 3)'s second argument.
        This value can be anything you choose.
        The statement above is only true for the hidden layers, not the last layer.

        Keras
            tf.keras.layers.Dense(256, activation='relu')
        """
        self.w3 = nn.Parameter(512, 256)
        self.b3 = nn.Parameter(1, 256)

        # Hidden layer 3
        """
        Parameter (weight 4)'s first argument must be the same for the previous parameter (bias 3)'s second argument.
        The statement above is only true for non input layers.

        Parameter (weight 4)'s second argument must be the input for the following parameter (bias 4)'s second argument.
        This value can be anything you choose.
        The statement above is only true for the hidden layers, not the last layer.

        Keras
            tf.keras.layers.Dense(124, activation='relu')
        """
        self.w4 = nn.Parameter(256, 124)
        self.b4 = nn.Parameter(1, 124)

        # Output layer
        """
        Parameter (weight 5)'s first argument must be the same for the previous parameter (bias 4)'s second argument.
        The statement above is only true for non input layers.

        Parameter (weight 5)'s second argument must be the size of the output which is 10 because of 10 digits

        Keras
            tf.keras.layers.Dense(1, activation='relu')
        """
        self.w5 = nn.Parameter(124, 10)
        self.b5 = nn.Parameter(1, 10)

        self.parameters = [self.w1, self.b1,
                           self.w2, self.b2,
                           self.w3, self.b3,
                           self.w4, self.b4,
                           self.w5, self.b5,
                           ]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        """
        Notes:
            From notes
                relu(x * w1 + b1) * W_2 + b_2

            What was created
                relu(relu(relu(x * w1 + b1) * W_2 + b_2) * W_3 + B_3) * w_4 + b_4
            
            This is pretty much a copy of Q2
        """
        # Input layer (Mimics hidden layer because why not)
        x_mul_w1 = nn.Linear(x, self.w1)
        x_mul_w1_add_b1 = nn.AddBias(x_mul_w1, self.b1)
        relu_1 = nn.ReLU(x_mul_w1_add_b1)
        layer_input = relu_1

        # Hidden Layer 1
        relu_1_mul_w2 = nn.Linear(layer_input, self.w2)
        relu_1_mul_w2_add_b2 = nn.AddBias(relu_1_mul_w2, self.b2)
        relu_2 = nn.ReLU(relu_1_mul_w2_add_b2)
        layer_hidden_1 = relu_2

        # Hidden Layer 2
        relu_2_mul_w3 = nn.Linear(layer_hidden_1, self.w3)
        relu_2_mul_w3_add_b3 = nn.AddBias(relu_2_mul_w3, self.b3)
        relu_3 = nn.ReLU(relu_2_mul_w3_add_b3)
        layer_hidden_2 = relu_3

        # Hidden Layer 3
        relu_3_mul_w4 = nn.Linear(layer_hidden_2, self.w4)
        relu_3_mul_w4_add_b4 = nn.AddBias(relu_3_mul_w4, self.b4)
        relu_4 = nn.ReLU(relu_3_mul_w4_add_b4)
        layer_hidden_3 = relu_4

        # Output Layer
        relu_4_mul_w5 = nn.Linear(layer_hidden_3, self.w5)
        relu_4_mul_w5_add_b5 = nn.AddBias(relu_4_mul_w5, self.b5)
        relu_5 = nn.ReLU(relu_4_mul_w5_add_b5)  # CANNOT USE THIS FOR layer_output
        layer_output = relu_4_mul_w5_add_b5

        return layer_output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        forward_pass = self.run(x)

        """
        We are doing classification so we proportionally distribute values as to what class
        the input is.
        
        Notes:
        
        Reference:
            Neural Networks Part 5: ArgMax and SoftMax
                https://www.youtube.com/watch?v=KpKog-L9veg
        
        """
        return nn.SoftmaxLoss(forward_pass, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        for _ in range(self.EPOCHS):

            # print(nn.Constant(dataset.x), nn.Constant(dataset.y))

            for x, y in dataset.iterate_once(self.batch_size):

                loss = self.get_loss(x, y)
                # print(loss)
                # print(nn.as_scalar(loss))

                gradient = nn.gradients(loss, self.parameters)
                # print(gradient)

                # Only works for the first gradient[0] because the shape I think...
                # print(nn.as_scalar(nn.DotProduct(gradient[0], gradient[0])))

                # Update parameters
                for index, parameter in enumerate(self.parameters):
                    parameter.update(gradient[index], self.learning_rate * -1)

            """
            Validation loss (pseudo). It is pseudo because the training dataset is the testing dataset.

            Notes:
                How far the model is from the actual true data
            """
            loss_val_pseudo = nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)))
            # print(loss_val_pseudo)

            if loss_val_pseudo < self.early_stopping_threshold:
                break

            """
            Validation data is available for this dataset. In this assignment, only the Digit Classification and 
            Language Identification datasets have validation data.

            Notes:
                Because we are doing images, a very small loss is very unlikely. So, we need to break
                via accuracy
            """
            print("Validation Accuracy: {}".format(dataset.get_validation_accuracy()))
            if dataset.get_validation_accuracy() > self.val_accuracy_sufficient_threshold:
                break

            """
            Reduce learning rate on plateau

            Notes:
                This is similar to Keras's ReduceLROnPlateau callback function, but
                it reduces the learning rate when the ratio between the old loss over the new loss abs value is less 
                than self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value
            """
            loss_val_new = abs(loss_val_pseudo)

            if self._loss_val_previous is not None:

                # Ratio between val loss and early stopping threshold (EST)
                ratio_val_loss_new_over_old_abs_value = abs(1 - (loss_val_new / self._loss_val_previous))

                # print(ratio_val_loss_new_over_old_abs_value)

                """
                If the amount of times already reduced is > 0 and
                If the ratio between the old loss over the new loss abs value is less than 
                self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value then reduce the learning rate 
                """
                if (self.reduce_learning_rate_amount_of_reductions > 0 and (
                        ratio_val_loss_new_over_old_abs_value < self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value)):
                    # print(self.learning_rate, self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value)

                    _learning_rate_old = self.learning_rate

                    # Reduce the learning rate
                    self.learning_rate *= self.reduce_learning_rate_multiple

                    # Reduce the threshold even more
                    self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value *= self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value_ratio

                    # Reduce the amount of time to reduce the learning rate
                    self.reduce_learning_rate_amount_of_reductions -= 1

                    print("Reducing learning rate from {} to {}".format(_learning_rate_old, self.learning_rate))
                    # print(self.learning_rate, self.reduce_learning_rate_threshold_loss_new_over_loss_old_abs_value)

            self._loss_val_previous = loss_val_new


"""
Question 4 (Not Required): Language Identification

Notes:

IMPORTANT NOTES:
    THIS QUESTION AS THE POTENTIAL TO FAIL WHILE TRAINING BECAUSE VALUES GET TO SMALL OR TO BIG

Reference:
    MIT 6.S191: Recurrent Neural Networks
        Reference:
            https://www.youtube.com/watch?v=qjrad0V0uJE

Result:
    ...
    Validation Accuracy: 0.888
    Your final test set accuracy is: 85.400000%
    *** PASS: check_lang_id
    
    ### Question q4: 7/7 ###
    
    Finished at 1:39:06
    
    Provisional grades
    ==================
    Question q4: 7/7
    ------------------
    Total: 7/7

"""


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        """
        Notes:
        
        """
        ####################
        """
        Special stuff
        """

        """
        This accuracy is used instead of loss and val loss because we are dealing with images
        and it's very difficult to get a good model to identify all images correctly

        Notes:
            To get max points for grading:
                To receive full points on this problem, your architecture should be able to achieve an accuracy of at 
                least 81% on the test set.
                
                Use a value greater than 81% because grading is weird for this problem
        """
        self.val_accuracy_sufficient_threshold = .88  # 88% is pretty high so training will take some time.

        ####################
        """
        Hyper Hyper Parameters
        """
        # Used for scaling the learning rate and batch size appropriately for faster training
        """
        
        Notes:
            YOU WANT THIS TO BE BIG BECAUSE IT WILL MAKE THE BATCH SIZE BIG AND LEARNING RATE BIG.
            Most importantly, you want teh batch size to be big so you don't get the vanishing gradient problem I think.
            
        """
        self.multiplier = 64

        ####################
        """
        Hyper Parameters

        """
        self.learning_rate = .001
        self.batch_size = 1  # This value is modified by self.multiplier
        self.EPOCHS = 100000  # There are always epochs (DON'T CHANGE THIS)

        ####################
        """
        Stuff the user should not see because it's meta stuff

        Notes:
            Basically, we aer multiplying a hyper parameter by a hyper hyper parameter

        Reference:
            15. Batch Size and Learning Rate in CNNs
                Notes:
                    " there's actually a subtle point there that it's important to know which is
                    that as you increase the batch size you should also increase the
                    learning rate so if you multiply the batch size by four you should also
                    generally speaking and multiply the learning rate by four so with bigger
                    batches because you're averaging over larger number samples you can get away
                    with higher learning rates than you otherwise could..."

                Reference:
                    https://www.youtube.com/watch?v=ZBVwnoVIvZk
        """
        self.learning_rate = self.learning_rate * self.multiplier
        self.batch_size = self.batch_size * self.multiplier

        ####################

        """
        Model

        Notes:            
            No Validation loss (pseudo)
            No Reduce learning rate on plateau
            No Early stopping
            
        IMPORTANT NOTES:

        """

        # Input layer
        """
        Parameter (weight 1)'s first argument must be the size of the input.

        Parameter (weight 1)'s second argument must be the input for the following parameter (bias 1)'s second 
        argument.
        This value can be anything you choose.
        The statement above is only true for the hidden layers, not the last layer.

        """
        self.w1 = nn.Parameter(self.num_chars, 168)
        self.b1 = nn.Parameter(1, 168)

        # Recurrent cell state
        """
        Parameter (weight )'s first argument must be the size of the input because self.w_t is a weight with respect to
        time.

        Parameter (weight 2)'s second argument must be the input for the following parameter 
        (weight of cell state h with respect to time t)'s second argument.
        This value can be anything you choose.

        """
        self.w_t = nn.Parameter(self.num_chars, 168)

        """
        Weight for h where h is the cell state. So self.w_h_t is weight for cell state h based on time t

        Notes:
            Parameter (self.w_h_t)'s first argument must be same as the second argument of the previous parameter
            (self.w_t).

            Parameter (self.w_h_t)'s second argument must be the same size as the first argument because this weight
            goes back into the same cell state (h) but at a different time (t).

        Reference:
            MIT 6.S191: Recurrent Neural Networks
                Notes:
                    h_t = f_w(x_t, h_t-1)

                        h_t = cell state
                        f_w = function with weights w
                        x_t = inputs
                        h_t-1 = old state (Past memory)

                    In this case, self.w_h_t is the weight

                Reference:
                    https://www.youtube.com/watch?v=qjrad0V0uJE&t=1125s

        """
        # 1st and 2nd parameters needs to be the same size because the output goes back into the cell again
        self.w_h_t = nn.Parameter(168, 168)

        # Bias with respect to time
        self.b_t = nn.Parameter(1, 168)

        # Output layer
        """
        Parameter (weight 3)'s first argument must be the same size as the second argument of the previous parameter 
        (Bias with respect to time).

        Parameter (weight 3)'s second argument must be the the same size as the output which is the amount of possible
        languages.
        """
        self.w3 = nn.Parameter(168, len(self.languages))
        self.b3 = nn.Parameter(1, len(self.languages))

        self.parameters = [
            # Weight and Bias for first character
            self.w1, self.b1,
            # Weight and Bias for the following characters with respect to the previous cell states except the last char
            self.w_t, self.w_h_t, self.b_t,
            # Weight and Bias for the last character
            self.w3, self.b3,
        ]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        char0 = xs[0]  # First char

        char0_mul_w1 = nn.Linear(char0, self.w1)
        char0_mul_w1_add_b1 = nn.AddBias(char0_mul_w1, self.b1)
        relu_1 = nn.ReLU(char0_mul_w1_add_b1)
        layer_input = relu_1
        h_0 = layer_input

        h_t = h_0  # h_9 (First h for the first character) will be the initial h_t

        # Recurrent Neural Network
        for character in xs[1:]:
            character_mul_w2 = nn.Linear(character, self.w_t)
            h_current = character_mul_w2

            h_previous = nn.Linear(h_t, self.w_h_t)

            h_current_add_h_previous = nn.Add(h_current, h_previous)

            h_i_mul_w1_add_b2 = nn.AddBias(h_current_add_h_previous, self.b_t)
            relu_i = nn.ReLU(h_i_mul_w1_add_b2)
            h_t = relu_i

        temp_mul_w2 = nn.Linear(h_t, self.w3)
        temp_mul_w2_add_b2 = nn.AddBias(temp_mul_w2, self.b3)
        layer_output = temp_mul_w2_add_b2

        return layer_output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        forward_pass = self.run(xs)

        return nn.SoftmaxLoss(forward_pass, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        for _ in range(self.EPOCHS):

            # print(nn.Constant(dataset.x), nn.Constant(dataset.y))

            for x, y in dataset.iterate_once(self.batch_size):

                loss = self.get_loss(x, y)
                # print(loss)
                # print(nn.as_scalar(loss))

                gradient = nn.gradients(loss, self.parameters)
                # print(gradient)

                # Only works for the first gradient[0] because the shape I think...
                # print(nn.as_scalar(nn.DotProduct(gradient[0], gradient[0])))

                # Update parameters
                for index, parameter in enumerate(self.parameters):
                    parameter.update(gradient[index], self.learning_rate * -1)

            """
            Validation data is available for this dataset. In this assignment, only the Digit Classification and 
            Language Identification datasets have validation data.

            """
            print("Validation Accuracy: {}".format(dataset.get_validation_accuracy()))
            if dataset.get_validation_accuracy() > self.val_accuracy_sufficient_threshold:
                break
