import numpy as np

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

class NeuralNetwork:
    MIN_WEIGHT = -1
    MAX_WEIGHT = 1
    MIN_BIAS = -1
    MAX_BIAS = 1

    ACTIVATION_FUNCTIONS = {
        "relu": lambda x : np.maximum(0, x),
        "sigmoid": lambda x : 1 / (1 + np.exp(-x)),
        "linear": lambda x : x,
        "tanh": lambda x : np.tanh(x),
        "softmax": softmax 
    }

    def __init__(self, layers_dimensions:list, hidden_activation_function:str = "relu", output_activation_function:str = "linear"):
        """
        Constructeur du réseau neuronal

        Parameters
        ----------
        layers_dimensions:list
            Dimensions de chaques layers sous forme de liste et comprenant input, hidden et output layer
            [1, 10, 50, 10, 2]
        """
        self.weights = [] 
        self.biases = []

        if not hidden_activation_function in self.ACTIVATION_FUNCTIONS or not output_activation_function in self.ACTIVATION_FUNCTIONS:
            raise ValueError("La fonction d'activation spécifiée n'existe pas.")
        self.activation_functions = [self.ACTIVATION_FUNCTIONS[hidden_activation_function] for _ in layers_dimensions[1:-1]] + [self.ACTIVATION_FUNCTIONS[output_activation_function]]

        for idx, neurons_numbers in enumerate(layers_dimensions[1:], start=1):
            w = np.random.uniform(self.MIN_WEIGHT, self.MAX_WEIGHT, (neurons_numbers, layers_dimensions[idx - 1]))
            b = np.random.uniform(self.MIN_BIAS, self.MAX_BIAS, (neurons_numbers, 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def prediction(self, inputs):
        values = np.array(inputs).reshape(-1, 1)

        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            print(values)
            res = np.dot(w, values) + b
            values = self.activation_functions[idx](res)
        
        return values.flatten().tolist()