import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten 


def dl_model1(input_shape: int, activation_function : str, no_of_output_classes : int, last_layer_activation = str) -> Model :
    """
    """
    num_classes = no_of_output_classes

    classifier = Sequential()

    classifier.add(Dense(5, kernel_initializer='uniform', activation = activation_function, input_dim = input_shape))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(3, kernel_initializer='uniform', activation = activation_function))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(num_classes, kernel_initializer='uniform', activation = last_layer_activation))

    return classifier


