import random
from typing import Optional
import sys
import numpy as np
from collections import defaultdict

from src.learning.neural_network import NeuralNetwork
from src.games.tictactoe import *

def board_1D(board, player=O):
    return [1 if cell == player else 0 if cell is BLANK else -1 for row in board for cell in row]

class Evolution:
    def __init__(self, n:int):
        """
        Classe Evolution permettant de sélectionner les meilleurs réseaux parmi n réseaux
        
        :param n: Nombre de réseaux total 
        :type n: int
        """
        self.n = n
        self.nns = [self._create_random_nn() for _ in range(n)]

    def train(self, epoch:int):
        """
        Sélectionne les meilleures IA dans un tournoi puis réitère
        
        :param epoch: Nombre d'itérations
        :type epoch: int
        """
        k = int(1/3 * self.n)

        for e in range(epoch):
            scores = self.tournament()
            self.nns.sort(key= lambda x: scores[x], reverse=True)

            for idx in range(k, int(self.n * 2/3)):
                self.nns[idx] = self._mutate(self.nns[idx % k])

            for idx in range(int(self.n * 2/3), self.n):
                self.nns[idx] = self._create_random_nn()

            sys.stdout.write(
                f"\rEpoch : {e}/{epoch} "
                f"Best score : {scores[self.nns[0]]}"
            )

    def tournament(self, n_matches:int = 10) -> list:
        """
        Renvoie les modèles du meilleur au pire par leurs performance dans un tournoi
        
        :param n_matches: Nombre de matchs joués par chaque modèles
        :type n_matches: int 
        :return: Liste des réseau triés du plus performant au pire
        :rtype: list
        """
        scores = defaultdict(int)

        for m in self.nns:
            opponents = random.sample(self.nns, n_matches)
            for o in opponents:
                if m is o:
                    continue
                if random.uniform(0, 1) < 0.5:
                    s = self.duel(m, o)
                else:
                    s = self.get_game_score(m, o)

                scores[m] += s * (5 if s < 0 else 1)
                scores[o] -= s * (5 if s > 0 else 1)
        
        return scores

    def duel(self, nn1, nn2):
        s1 = self.get_game_score(nn1, nn2)
        s2 = self.get_game_score(nn2, nn1)
        return s1 - s2

    def get_game_score(self, nn1, nn2, display = False, early_stop = False):
        game = Board()

        players = {X: nn1, O: nn2}
        coups = 0

        while not game.is_ended():
            if early_stop:
                if coups >= 5 and game.winner is None:
                    return 0
            
            current_player = game.get_current_player()
            nn = players[current_player]

            move = self._get_move(game, nn)

            try:
                game[move] = current_player
            except Exception:
                return -0.1 if nn is nn1 else 0.1
            
            coups += 1
            
            if display:
                print(game, end="\n\n")

        if game.winner is None:
            return 0.2

        winner_nn = players[game.winner]
        get_score = lambda x: x - coups/10
        return get_score(1 if winner_nn is nn1 else -1)
    
    def _create_random_nn(self) -> NeuralNetwork:
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
            output_activation_function="linear"
        )

        return nn

    def _get_move(self, board: Board, nn: NeuralNetwork):
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
    
    def _mutate(self, nn: NeuralNetwork,
            sigma=0.02) -> NeuralNetwork:
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

def main(): 
    evol = Evolution(75)
    evol.train(200)
    evol.get_game_score(evol.nns[0], evol.nns[1], True, False)