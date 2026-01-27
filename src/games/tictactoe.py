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

        if self._get_horizontal_win(board) or self._get_vertical_win(board) or self._get_diagonal_win(board):
            return True
        
        if not any([any([val is BLANK for val in row]) for row in self]):
            return True
        
        return False
    
    def get_legal_move(self, board = None) -> list[Position]:
        if board is None:
            board = self
        
        res = []
        for y, row in enumerate(board):
            for x, val in enumerate(row):
                if val is BLANK:
                    res.append(Position(x, y))
        
        return res

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
            self.winner = board[0][2]
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
        
    def __hash__(self):
        return super().__hash__()
        
def rotate90(board:Board):
    return [list(row) for row in zip(*board)][::-1]

def rotate_move_90(pos:Position, n=3):
    return Position(pos.y, n - 1 - pos.x)

def mirror_h(board):
    return [row[::-1] for row in board]

def mirror_move(pos, n=3):
    return Position(n - 1 - pos.x, pos.y)

def data_augmentation(board: Board, move: Position):
    cur = board
    cur_move = move

    for _ in range(4):
        yield cur, cur_move
        yield mirror_h(cur), mirror_move(cur_move)

        cur = rotate90(cur)
        cur_move = rotate_move_90(cur_move)

b = Board()

assert not b.is_ended()

if __name__ == "__main__":
    b[Position(0, 0)] = X
    for board, move in data_augmentation(b, Position(0, 0)):
        print(Board(board), move)
        input()