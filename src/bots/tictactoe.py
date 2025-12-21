import random
from typing import Optional
import sys
import numpy as np

from src.learning.neural_network import NeuralNetwork
from src.games.tictactoe import *

def get_move(board: Board, nn: NeuralNetwork):
    moves_prob = nn.prediction(board_1D(board))

    legal_moves = board.get_legal_move()

    best_move = None
    best_score = -float("inf")

    for pos in legal_moves:
        idx = pos.x * 3 + pos.y
        score = moves_prob[idx]

        if score > best_score:
            best_score = score
            best_move = pos

    return best_move

def get_result(nn1, nn2, display = False):
    game = Board()

    if random.randint(0, 1) == 0:
        players = {X: nn1, O: nn2}
    else:
        players = {X: nn2, O: nn1}

    while not game.is_ended():
        current_player = game.get_current_player()
        nn = players[current_player]

        move = get_move(game, nn)

        try:
            game[move] = current_player
        except Exception:
            return -2 if nn is nn1 else 2
        
        if display:
            print(game, end="\n\n")

    if game.winner is None:
        return 0

    winner_nn = players[game.winner]
    return 1 if winner_nn is nn1 else -1

def create_random_nn() -> NeuralNetwork:
    min_layers = 1
    max_layers = 3

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

def mutate(nn: NeuralNetwork,
           sigma=0.5) -> NeuralNetwork:
    """
    p_weight : proba de muter les poids
    sigma    : amplitude du bruit
    """

    weights = [w.copy() for w in nn.weights]
    biases = [b.copy() for b in nn.biases]

    for i in range(len(weights)):
        weights[i] += np.random.normal(0, sigma, weights[i].shape)
        biases[i]  += np.random.normal(0, sigma, biases[i].shape)

    return NeuralNetwork(
        weights=weights,
        biases=biases,
        hidden_activation_function=nn.hidden_activation_function,
        output_activation_function=nn.output_activation_function
    )

def board_1D(board, player=O):
    return [1 if cell == player else 0 if cell is BLANK else -1 for row in board for cell in row]

def main():
    nn = create_random_nn()
    nn2 = create_random_nn()

    wins_1 = 0
    wins_2 = 0
    draws = 0
    games = 0

    for epoch in range(10000):
        #print(epoch)
        score = get_result(nn, nn2)

        if score == -2 or score == 2:
            if score == -2:
                nn = create_random_nn()
            else:
                nn2 = create_random_nn()
            continue

        games += 1

        if score == 1:
            wins_1 += 1
            nn = nn
            nn2 = mutate(nn)
        elif score == -1:
            wins_2 += 1
            nn = nn2
            nn2 = mutate(nn)
        else:
            draws += 1

        p1 = wins_1 / games * 100
        p2 = wins_2 / games * 100
        pd = draws / games * 100

        bar = lambda p: "█" * int(p / 5)

        sys.stdout.write(
            f"\rGAMES {games}/{epoch} | "
            f"NN1 [{bar(p1):20}] {p1:5.1f}% | "
            f"NN2 [{bar(p2):20}] {p2:5.1f}% | "
            f"Draws {pd:5.1f}%"
        )
        sys.stdout.flush()
    
    get_result(nn, nn2, True)