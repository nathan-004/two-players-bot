from typing import Optional
from copy import deepcopy
from typing import NamedTuple

X_I = "X" # X interface
O_I = "O" # O Interface
X = 0 # X stockage
O = 1 # O stockage
BLANK = None # Vide stockage
BLANK_I = " " # Vide affichage

class Position(NamedTuple):
    x:int
    y:int

class AlreadyTaken(Exception):
    pass

def blank_board() -> list:
    return [Row([BLANK for x in range(3)]) for y in range(3)]

class Row(list):
    def __str__(self):
        return " | ".join([X_I if val == X else O_I if val == O else BLANK_I for val in self])

    def __setitem__(self, idx, val):
        if not self[idx] is BLANK:
            raise AlreadyTaken("Case déjà utilisée")
        super().__setitem__(idx, val)

class Board(list):
    """
    Contain the rules of TicTacToe
    """

    # -------------------------------------------------------
    # Utils Functions                                       |
    # -------------------------------------------------------

    def get_current_player(self, board = None):
        if board is None:
            board = self

        t = 0
        for row in board:
            for val in row:
                if not val is BLANK:
                    t += 1
        return X if t % 2 == 0 else O

    def is_ended(self, board = None) -> int:
        """Renvoie True si fin du jeu et modifies self.winner si vainqueur"""
        if board is None:
            board = self

        if not any([any([val is BLANK for val in row]) for row in self]):
            return True

        if self._get_horizontal_win(board) or self._get_vertical_win(board) or self._get_diagonal_win(board):
            return True
        
        return False

    def _get_horizontal_win(self, board = None):
        for row in board:
            w = all([row[x] == row[x+1] and not row[x] is BLANK for x in range(2)])
            if w:
                self.winner = row[0]
                return True
        return False

    def _get_vertical_win(self, board=None):
        for x in range(3):
            w = all([board[y][x] == board[y+1][x] and not board[y][x] is BLANK for y in range(2)])
            if w:
                self.winner = board[0][x]
                return True
        return False

    def _get_diagonal_win(self, board=None):
        if board[0][0] == board[1][1] == board[2][2] and not board[0][0] is BLANK:
            self.winner = board[0][0]
            return True
        if board[0][2] == board[1][1] == board[2][0] and not board[0][2] is BLANK:
            self.winner = board[0][0]
            return True
        
        return False

    # -------------------------------------------------------
    # Special Functions                                     |
    # -------------------------------------------------------

    def __init__(self, board:Optional[list] = None):
        if board is None:
            board = blank_board()
        else:
            board = deepcopy([Row(row) for row in board])
        self.winner = BLANK
        super().__init__(board)

    def __str__(self):
        return "\n".join([row.__str__() for row in self])

    def __setitem__(self, idx, val):
        if isinstance(idx, Position):
            self[idx.y][idx.x] = val
        elif isinstance(idx, tuple):
            self[idx[1]][idx[0]] = val
        else:
            super().__setitem__(idx, val)

    def __getitem__(self, idx):
        if isinstance(idx, Position):
            return self[idx.y][idx.x]
        else:
            return super().__getitem__(idx)

b = Board()
assert not b.is_ended()