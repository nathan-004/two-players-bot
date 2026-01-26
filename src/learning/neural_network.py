import numpy as np
import os
import pickle
from typing import Optional

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

class NeuralNetwork:
    MIN_WEIGHT = -10.0
    MAX_WEIGHT = 10.0
    MIN_BIAS   = -5
    MAX_BIAS   = 5

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
    
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarde le réseau neuronal dans un fichier via pickle.
        Le fichier contient un dict avec les clés: 'weights', 'biases',
        'hidden_activation_function' et 'output_activation_function'.
        """
        data = {
            "weights": self.weights,
            "biases": self.biases,
            "hidden_activation_function": self.hidden_activation_function,
            "output_activation_function": self.output_activation_function,
        }

        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    
    
    @classmethod
    def load(cls, filepath: str) -> "NeuralNetwork":
        """
        Charge un réseau neuronal depuis un fichier et retourne une instance de NeuralNetwork.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        # validation minimale
        if not {"weights", "biases"}.issubset(data):
            raise ValueError("Fichier de réseau invalide (clés manquantes).")
        return cls.from_weights(weights=data["weights"], biases=data["biases"],
                                 hidden_activation_function=data.get("hidden_activation_function", "relu"),
                                 output_activation_function=data.get("output_activation_function", "linear"))
    

def save(network: NeuralNetwork, filepath: str) -> None:
    """
    Fonction utilitaire: sauvegarde un NeuralNetwork sur disque.
    """
    network.save(filepath)
    
    
def load(filepath: str) -> NeuralNetwork:
    """
    Fonction utilitaire: charge un NeuralNetwork depuis un fichier.
    """
    return NeuralNetwork.load(filepath)