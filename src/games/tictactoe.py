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

class Board(list):
    """
    Contain the rules of TicTacToe
    """

    # -------------------------------------------------------
    # Special Functions                                     |
    # -------------------------------------------------------

    def __init__(self, board:Optional[list] = None):
        if board is None:
            board = blank_board()
        super().__init__(deepcopy(board))

    def __str__(self):
        return "\n".join([row.__str__() for row in self])
    
    def __setitem__(self, idx, val):
        if isinstance(idx, Position):
            if self[idx] is BLANK:
                raise AlreadyTaken("Case déjà utilisée")
            self[idx.y][idx.x] = val
        else:
            super().__setitem__(idx, val)
    
    def __getitem__(self, idx):
        if isinstance(idx, Position):
            return self[idx.y][idx.x]
        else:
            return super().__getitem__(idx)

a = Board()
a[Position(0,0)] = 1
print(a)