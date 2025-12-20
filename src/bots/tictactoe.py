import random
from typing import Optional
import sys

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
        cur = o if game.get_current_player() == O else x
        move = get_move(game, cur)
        try:
            game[move] = game.get_current_player()
        except Exception as e:
            return {O: 1, X: -1}[game.get_current_player()]
    
    return {O: -1, X: 1, None: 0}[game.winner]

def create_random_nn() -> NeuralNetwork:
    input_size = 10
    output_size = 9

    # hyperparamètres évolutifs
    min_layers = 1
    max_layers = 3
    min_neurons = 16
    max_neurons = 64

    n_hidden = random.randint(min_layers, max_layers)

    hidden_sizes = [
        random.choice([16, 24, 32, 48, 64])
        for _ in range(n_hidden)
    ]

    nn = NeuralNetwork(
        layers_dimensions=[9] + hidden_sizes + [9],
        hidden_activation_function="relu",
        output_activation_function="softmax"
    )

    return nn

def board_1D(board, player=O):
    return [1 if cell == player else 0 if cell is BLANK else -1 for row in board for cell in row]

def main():
    nn = create_random_nn()
    nn2 = create_random_nn()

    wins_1 = 0
    wins_2 = 0
    draws = 0
    games = 0

    for epoch in range(500000):
        #print(epoch)
        score = get_result(nn, nn2)

        if score is None:
            continue

        games += 1

        if score == 1:
            wins_1 += 1
            nn = nn
            nn2 = create_random_nn()
        elif score == -1:
            wins_2 += 1
            nn = nn2
            nn2 = create_random_nn()
        else:
            draws += 1

        p1 = wins_1 / games * 100
        p2 = wins_2 / games * 100
        pd = draws / games * 100

        bar = lambda p: "█" * int(p / 5)

        sys.stdout.write(
            f"\rGAMES {games} | "
            f"NN1 [{bar(p1):20}] {p1:5.1f}% | "
            f"NN2 [{bar(p2):20}] {p2:5.1f}% | "
            f"Draws {pd:5.1f}%"
        )
        sys.stdout.flush()