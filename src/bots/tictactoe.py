import random
from typing import Optional

from src.learning.neural_network import NeuralNetwork
from src.games.tictactoe import *

def create_random_nn() -> NeuralNetwork:
    nn = NeuralNetwork(
        layers_dimensions=[9, 128, 128, 64, 9],
        hidden_activation_function="relu",
        output_activation_function="softmax"
    )

    return nn

def board_1D(board, player=O):
    return [1 if cell == player else 0 if cell is BLANK else -1 for row in board for cell in row]

def main():
    nn = create_random_nn()