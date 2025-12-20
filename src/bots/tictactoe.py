import random
from typing import Optional

from src.learning.neural_network import NeuralNetwork
from src.games.tictactoe import *

def get_move(board:Board, nn:NeuralNetwork):
    moves_prob = nn.prediction(board_1D(board))

    max_idx, max_val = 0,0,
    for idx, el in enumerate(moves_prob):
        if el > max_val:
            max_idx, max_val = idx, el
        if el > 0.5:
            break

    return divmod(max_idx, 3)

def get_result(neural_network1, neural_network2):
    game = Board()
    i = random.randint(0, 1)
    o = (neural_network1, neural_network2)[i]
    x = neural_network1 if i == 1 else neural_network2

    while not game.is_ended():
        print(game, end="\n\n")

        cur = o if game.get_current_player() == O else x
        move = get_move(game, cur)
        try:
            game[move] = game.get_current_player()
        except Exception as e:
            return {O: 1, X: -1}[game.get_current_player()]
    
    return {O: -1, X: 1, None: 0}[game.winner]

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
    nn2 = create_random_nn()
    print(get_result(nn, nn2))