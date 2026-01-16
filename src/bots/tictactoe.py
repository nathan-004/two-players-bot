import random
from typing import Optional
import sys
import numpy as np
from collections import defaultdict
from copy import deepcopy

from src.learning.neural_network import NeuralNetwork, save
from src.games.tictactoe import *

def board_1D(board, player=O):
    return [1 if cell == player else 0 if cell is BLANK else 0 for row in board for cell in row]

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

    def get_game_score(self, nn1, nn2, display = False, early_stop = False, return_first_move:bool = False):
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
                first_move = move
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
            s = 0.5
        elif players[game.winner] is nn1:
            s = 1.0
        else:
            s = 0
        
        if not return_first_move:
            return s
        else:
            return (s, first_move)
    
    def _create_random_nn(self, hidden_sizes:list = None, n_board:int = 2) -> NeuralNetwork:
        if hidden_sizes is None:
            min_layers = 1
            max_layers = 3

            n_hidden = random.randint(min_layers, max_layers)

            hidden_sizes = [
                random.choice([4, 8, 16, 24, 32, 48, 64])
                for _ in range(n_hidden)
            ]

        nn = NeuralNetwork(
            layers_dimensions=[9*n_board] + hidden_sizes + [9],
            hidden_activation_function="relu",
            output_activation_function="softmax"
        )

        return nn

    def _get_move(self, board: Board, nn: NeuralNetwork, allow_illegal_moves = False):
        if not type(nn) is NeuralNetwork:
            return nn._get_move(board)

        p = board.get_current_player()
        moves_prob = nn.prediction(board_1D(board, p) + board_1D(board, {X: O, O: X}[p]))

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
    
    def _targeted_mutation(self, nn:NeuralNetwork, board:Board, player:int, last_move:Position, sigma = 0.01, factor:float = 1.0):
        """
        Modifie les poids et les biais en fonction de la part d'activation de chaque neurone vers le coups perdant

        :param last_move: Dernier coups joué avant une défaite
        :type last_move: Position
        :param sigma: Defini intervalle d'aléatoire de la mutation
        """
        res = self._get_index_from_move(last_move)

        weights = [w.copy() for w in nn.weights]
        biases = [b.copy() for b in nn.biases]

        # activations stored as 1D arrays (matching np.dot(w, a) semantics)
        activations = [np.array(board_1D(board, player) + board_1D(board, {X: O, O: X}[player]), dtype=float)]

        # Stocker les activations (forward pass)
        for idx, (w, b) in enumerate(zip(weights, biases)):
            # w: (neurons, prev), b: (neurons, 1)
            z = np.dot(w, activations[-1]) + b.flatten()
            a = nn.activation_functions[idx](z)
            activations.append(a)

        # Calcul des responsabilités (1D arrays)
        responsibilities = [np.zeros_like(a) for a in activations]
        responsibilities[-1][res] = 1.0

        # Retro Propagation des responsabilités
        for l in reversed(range(len(weights))):
            w = weights[l]
            r_next = responsibilities[l + 1]

            # w.T: (prev, next), r_next: (next,)
            contrib = np.abs(w.T * r_next).sum(axis=1)
            responsibilities[l] = contrib / (contrib.sum() + 1e-8)

        new_weights = []
        new_biases = []

        for l, (w, b) in enumerate(zip(weights, biases)):
            r_out = responsibilities[l + 1][:, None]  # (next,1)
            r_in = responsibilities[l][None, :]       # (1, prev)
            factor_ = r_out * r_in                     # (next, prev)

            noise_w = np.random.normal(0, sigma, w.shape) * factor_
            noise_b = np.random.normal(0, sigma, b.shape) * r_out

            new_weights.append(w + factor * np.absolute(noise_w))
            new_biases.append(b + factor * np.absolute(noise_b))

        return NeuralNetwork(
            weights=new_weights,
            biases=new_biases,
            hidden_activation_function=nn.hidden_activation_function,
            output_activation_function=nn.output_activation_function
        )

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
            opponents = random.sample(self.nns, n_matches) # self._get_opponents(m, n_matches) 
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
        self.hidden_sizes = hidden_sizes
        self.nn = self._create_random_nn(hidden_sizes)

    def train(self, epochs:int, sigma:float = 0.05, verbose:bool = True):
        """Entraine l'IA en jouant contre elle-même.

        Lors d'une défaite, applique une mutation ciblée sur le dernier coup joué
        par le joueur perdant pour réduire la probabilité de rejouer ce coup.

        :param epochs: Nombre de parties jouées
        :param sigma: Amplitude de la mutation ciblée
        :param verbose: Si True affiche la progression
        """
        games = {
            X: 0,
            O: 0,
            None: 0
        }
        
        opponent = self._mutate(self.nn) # Adversaire qui n'est pas modifié à chaque epoch
        cons = 0
        current_cons = BLANK

        for e in range(epochs):
            if cons >= 50:
                opponent = self._mutate(self.nn, sigma = 0.1)
            elif e % 100 == 0:
                opponent = self._mutate(self.nn, sigma=0.01)

            game = Board()
            history = []  # tuples (board_before_move, player, move)
            main_player = random.choice([X, O])

            while not game.is_ended():
                current_player = game.get_current_player()
                board_before = Board(game)  # copie de l'état avant le coup
                move = self._get_move(game, self.nn if current_player == main_player else opponent, allow_illegal_moves=False)
                history.append((board_before, current_player, move))

                try:
                    game[move] = current_player
                except Exception:
                    # Coup illégal -> l'autre joueur gagne
                    game.winner = O if current_player == X else X
                    break
                
            games[game.winner] += 1

            if verbose:
                total = max(1, games[X] + games[O] + games[None])

                print(
                    f"\rEpoch {e}/{epochs} | "
                    f"X: {games[X]:4d} ({games[X]/total:.1%})  "
                    f"O: {games[O]:4d} ({games[O]/total:.1%})  "
                    f"Draw: {games[None]:4d} ({games[None]/total:.1%})",
                    end="",
                    flush=True
                )
            if game.winner is BLANK and current_cons is BLANK:
                cons += 1
            elif game.winner == current_cons:
                cons += 1
            else:
                cons = 1
            
            current_cons = game.winner

            # Si match nul, on ne fait rien
            if game.winner is None:
                continue

            cons_draws = 0

            losing_player = X if game.winner == O else O
            winning_player = O if game.winner == O else X

            if losing_player == main_player:
                last_board = None
                last_move = None
                for b, p, mv in reversed(history):
                    if p == losing_player:
                        last_board = b
                        last_move = mv
                        break

                if last_move is not None:
                    # Appliquer une mutation ciblée pour diminuer l'activation du coup perdant
                    self.nn = self._targeted_mutation(self.nn, last_board, losing_player, last_move, sigma=sigma, factor=-1)

            if winning_player == main_player:
                last_board = None
                last_move = None
                for b, p, mv in reversed(history):
                    if p == winning_player:
                        last_board = b
                        last_move = mv
                        break

                if last_move is not None:
                    self.nn = self._targeted_mutation(self.nn, last_board, winning_player, last_move, sigma=sigma)

        if verbose:
            print("\nTraining completed.")

    def evaluate(self, n=5):
        total = 0
         
        for i in range(n):
            main_player = random.choice([X, O])
            total += self.get_game_score(self.nn, PerfectBot())
        
        return total / n
    
    def heatmap(self, nn = None, n = 30):
        if nn is None:
            nn = self.nn
        total = defaultdict(lambda : [0, 0]) # Dictionnaire position: score moyen
        
        for i in range(n):
            main_player = random.choice([X, O])
            score, first_move = self.get_game_score(nn, PerfectBot(), return_first_move = True)
            total[first_move][1] += score
            total[first_move][0] += 1
        
        for y in range(3):
            row = []
            for x in range(3):
                el = total[Position(x, y)]
                res = el[1] / max(el[0], 1)
                row.append(f"{res*100:.1f}")
            print(" | ".join(row))
        
def self_evol_main():
    """Lance un entraînement en self-play puis évalue le réseau contre `PerfectBot`."""
    evol = SelfEvolution([32, 64, 32])

    rounds = 4
    epochs_per_round = 5000
    best_score = 0
    best_nn = None

    for i in range(rounds):
        print(f"\n=== Round {i + 1}/{rounds} - training {epochs_per_round} epochs ===")
        evol.train(epochs_per_round, sigma=0.03, verbose=True)

        # Évaluer face au PerfectBot
        n = int(10 * (i+1)/5)
        score = evol.evaluate(n = n)
        if score >= best_score:
            best_score = score
            best_nn = deepcopy(evol.nn)
        print(f"Winrate vs perfect (X side): {score:.3f}, Best Winrate = {best_score}")
    
    evol.heatmap(best_nn)
    
    # Afficher une partie finale
    print("\nPartie finale (affichée) :")
    while input("press q to quit") != "q":
        evol.get_game_score(best_nn, PerfectBot(), display=True, early_stop=False)
    
    save(best_nn, "saves/nn1.pkl")

def main():
    self_evol_main()
