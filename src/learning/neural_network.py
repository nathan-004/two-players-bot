import numpy as np
from typing import Optional

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

class NeuralNetwork:
    MIN_WEIGHT = -1.0
    MAX_WEIGHT = 1.0
    MIN_BIAS   = -0.5
    MAX_BIAS   = 0.5

    ACTIVATION_FUNCTIONS = {
        "relu": lambda x : np.maximum(0, x),
        "sigmoid": lambda x : 1 / (1 + np.exp(-x)),
        "linear": lambda x : x,
        "tanh": lambda x : np.tanh,
        "softmax": softmax 
    }

    @classmethod
    def from_layers(cls, layers_dimensions, **kwargs):
        return cls(layers_dimensions=layers_dimensions, **kwargs)

    @classmethod
    def from_weights(cls, weights, biases, **kwargs):
        return cls(weights=weights, biases=biases, **kwargs)

    def __init__(self, layers_dimensions:Optional[list] = None, weights:Optional[list[np.ndarray]] = None, biases:Optional[list[np.ndarray]] = None, hidden_activation_function:str = "relu", output_activation_function:str = "linear"):
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

        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function

        if not hidden_activation_function in self.ACTIVATION_FUNCTIONS or not output_activation_function in self.ACTIVATION_FUNCTIONS:
            raise ValueError("La fonction d'activation spécifiée n'existe pas.")

        if not weights is None and not biases is None:
            self.weights =  weights
            self.biases = biases
            layers_dimensions = self._infer_layers_from_weights()
        elif not layers_dimensions is None: 
            for idx, neurons_numbers in enumerate(layers_dimensions[1:], start=1):
                w = np.random.uniform(self.MIN_WEIGHT, self.MAX_WEIGHT, (neurons_numbers, layers_dimensions[idx - 1]))
                b = np.random.uniform(self.MIN_BIAS, self.MAX_BIAS, (neurons_numbers, 1))
                self.weights.append(w)
                self.biases.append(b)
        else:
            raise ValueError("Argument layers_dimensions ou weights et biases manquant")
        self.activation_functions = [self.ACTIVATION_FUNCTIONS[hidden_activation_function] for _ in layers_dimensions[1:-1]] + [self.ACTIVATION_FUNCTIONS[output_activation_function]]
    
    def prediction(self, inputs):
        values = np.array(inputs).reshape(-1, 1)

        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            res = np.dot(w, values) + b
            values = self.activation_functions[idx](res)
        
        return values.flatten().tolist()
    
    def _infer_layers_from_weights(self):
        layers = [self.weights[0].shape[0]]
        for w in self.weights:
            layers.append(w.shape[1])
        return layers