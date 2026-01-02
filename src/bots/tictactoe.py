import random
from typing import Optional
import sys
import numpy as np
from collections import defaultdict

from src.learning.neural_network import NeuralNetwork
from src.games.tictactoe import *

def board_1D(board, player=O):
    return [1 if cell == player else 0 if cell is BLANK else -1 for row in board for cell in row]

def elo_update(r1, r2, s1, K=20):
    e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
    delta = K * (s1 - e1)
    return delta, -delta

class PlayerPlay:
    def _get_move(self, board:Board):
        move = False

        while not move:
            move = input("Coups sous format 'x y' : ")
            move = self._is_legal_move(move, board)
        return move
    
    def _is_legal_move(self, inp:str, board:Board) -> bool:
        if not " " in inp:
            return False
        
        els = inp.split(" ")

        if len(els) != 2:
            return False
        
        if not(all([el.isdigit() for el in els])):
            return False
        
        move = Position(int(els[0]), int(els[1]))

        if move in board.get_legal_move():
            return move 
        return False

class PerfectBot:
    def _get_move(self, board:Board):
        """Return the best `Position` for the current player using minimax."""
        score, move = self._get_best_move(board)
        return move

    def _get_best_move(self, board:Board):
        """Minimax algorithm returning a tuple (score, best_move).

        Score is from X's perspective: X win -> 1, O win -> -1, draw -> 0.
        """
        if board.is_ended():
            return ({X: 1, O: -1, BLANK: 0}[board.winner], None)

        current_player = board.get_current_player()

        if current_player == X:
            best_score = -float("inf")
        else:
            best_score = float("inf")

        best_move = None

        for move in board.get_legal_move():
            new_board = Board(board)
            new_board[move] = current_player
            score, _ = self._get_best_move(new_board)

            if current_player == X:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_score, best_move

class TicTacToeEvolution:
    def duel(self, nn1, nn2):
        s1 = self.get_game_score(nn1, nn2)
        s2 = 1 - self.get_game_score(nn2, nn1)
        return (s1 + s2) / 2

    def get_game_score(self, nn1, nn2, display = False, early_stop = False):
        game = Board()
        players = {X: nn1, O: nn2}
        
        coups = 0

        while not game.is_ended():
            if coups >= 5 and early_stop:
                return 0.5
            current_player = game.get_current_player()
            if coups >= 1:
                nn = players[current_player]
                move = self._get_move(game, nn, allow_illegal_moves=False)
            else:
                move = random.choice(game.get_legal_move())
            
            try:
                game[move] = current_player
            except Exception as e:
                #print(e)
                game.winner = 0 if current_player == 1 else 1
                break

            if display:
                print(game, end="\n\n")

            coups += 1

        if game.winner is None:
            return 0.5
        elif players[game.winner] is nn1:
            return 1.0
        else:
            return 0
    
    def _create_random_nn(self, hidden_sizes:list = None) -> NeuralNetwork:
        if hidden_sizes is None:
            min_layers = 1
            max_layers = 3

            n_hidden = random.randint(min_layers, max_layers)

            hidden_sizes = [
                random.choice([4, 8, 16, 24, 32, 48, 64])
                for _ in range(n_hidden)
            ]

        nn = NeuralNetwork(
            layers_dimensions=[9] + hidden_sizes + [9],
            hidden_activation_function="linear",
            output_activation_function="softmax"
        )

        return nn

    def _get_move(self, board: Board, nn: NeuralNetwork, allow_illegal_moves = False):
        if not type(nn) is NeuralNetwork:
            return nn._get_move(board)

        p = board.get_current_player()
        moves_prob = nn.prediction(board_1D(board, p))

        if allow_illegal_moves:
            max_ = max(moves_prob)
            idx = moves_prob.index(max_)
            x, y = divmod(idx, 3)
            return Position(x, y)

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
        sigma=0.003) -> NeuralNetwork:
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
    
    def _get_index_from_move(self, pos:Position):
        return pos.x * 3 + pos.y

class TournamentEvolution(TicTacToeEvolution):
    def __init__(self, n:int):
        """
        Classe Evolution permettant de sélectionner les meilleurs réseaux parmi n réseaux
        
        :param n: Nombre de réseaux total 
        :type n: int
        """
        self.n = n
        self.nns = [self._create_random_nn() for _ in range(n)]
        self.ratings = defaultdict(lambda : 1000)

    def train(self, epoch:int):
        """
        Sélectionne les meilleures IA dans un tournoi puis réitère
        
        :param epoch: Nombre d'itérations
        :type epoch: int
        """
        k = int(0.3 * self.n)

        for e in range(epoch):
            scores = self.tournament()
            self.nns.sort(key= lambda x: scores[x], reverse=True)

            for idx in range(k, int(self.n * 0.5)):
                self.nns[idx] = self._mutate(self.nns[idx % k], sigma=0.0001)

            for idx in range(int(self.n * 0.5), int(self.n * 0.8)):
                self.nns[idx] = self._mutate(self.nns[idx % k], sigma=0.05)

            for idx in range(int(self.n * 0.8), self.n):
                self.nns[idx] = self._create_random_nn()

            sys.stdout.write(
                f"\rEpoch : {e}/{epoch} "
                f"Best elo : {scores[self.nns[0]]} "
                f"Worst elo : {scores[self.nns[-1]]}"
            )

            sys.stdout.flush()

    def tournament(self, n_matches:int = 1) -> list:
        """
        Renvoie les modèles du meilleur au pire par leurs performance dans un tournoi
        
        :param n_matches: Nombre de matchs joués par chaque modèles
        :type n_matches: int 
        :return: Liste des réseau triés du plus performant au pire
        :rtype: list
        """
        if n_matches is None:
            n_matches = self.n

        scores = self.ratings

        for m in self.nns:
            opponents = self._get_opponents(m, n_matches)  #random.sample(self.nns, n_matches)
            for o in opponents:
                if m is o:
                    continue
                if random.uniform(0, 1) < 0.5:
                    s = self.duel(m, o)
                else:
                    s = self.get_game_score(m, o)

                delta_m, delta_o = elo_update(
                    scores[m], scores[o], s
                )
                scores[m] += delta_m
                scores[o] += delta_o
        
        self.ratings = scores
        return scores
    
    def _get_opponents(self, nn, n_opponents) -> list:
        """
        Renvoie la liste des meilleurs adversaires en fonction de l'elo de nn
        
        :param nn: Réseau neuronal de base
        :param n_opponents: Nombre d'adversaires
        :return: Liste de NeuralNetwork
        :rtype: list
        """
        elo_nn = self.ratings[nn]

        sorted_nns = sorted(
            (other for other in self.nns if other is not nn),
            key=lambda other: abs(self.ratings[other] - elo_nn)
        )

        return sorted_nns[:n_opponents]

def tournament_main(): 
    evol = TournamentEvolution(75)
    
    for i in range(10):
        evol.train(500)

        w = 0
        for _ in range(10):
            w += evol.get_game_score(evol.nns[0], PerfectBot())
        print(" Winrate vs perfect (X side) :", w / 10)
    print(evol.ratings[evol.nns[0]])
    evol.get_game_score(evol.nns[0], evol.nns[0], True, False)

class SelfEvolution(TicTacToeEvolution):
    """
    S'améliore en jouant répétivement contre lui même
    """

    def __init__(self, hidden_sizes = [16, 32, 16]):
        self.nn = self._create_random_nn(hidden_sizes)

    def train(self, epochs:int):
        
        for e in range(epochs):
            pass
    
    def _targeted_mutation(self, nn:NeuralNetwork, last_move:Position, sigma = 0.01):
        """
        Modifie les poids et les biais en fonction de la part d'activation de chaque neurone vers le coups perdant
        
        :param last_move: Dernier coups joué avant une défaite
        :type last_move: Position
        :param sigma: Defini intervalle d'aléatoire de la mutation
        """
        res = self._get_index_from_move(last_move)

        weights = [w.copy() for w in nn.weights]
        biases = [b.copy() for b in nn.biases]

        for idx in range(len(weights)):
            pass

def main():
    tournament_main()