from typing import Optional
from copy import deepcopy

X_I = "X" # X interface
O_I = "O" # O Interface
X = 0 # X stockage
O = 1 # O stockage
BLANK = None # Vide stockage
BLANK_I = " " # Vide affichage

def blank_board() -> list:
    return [Row([BLANK for x in range(3)]) for y in range(3)]

class Row(list):
    def __str__(self):
        return " | ".join([X_I if val == X else O_I if val == O else BLANK_I for val in self])

class Board(list):
    def __init__(self, board:Optional[list] = None):
        if board is None:
            board = blank_board()
        super().__init__(deepcopy(board))

    def __str__(self):
        return "\n".join([row.__str__() for row in self])
    
a = Board()
print(a)