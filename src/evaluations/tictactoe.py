import sqlite3
import json
from collections import deque

from src.games.tictactoe import *

def board_to_str(board:Board):
    res = json.dumps(board)
    return res

def str_to_board(chrs:str) -> Board:
    lst = json.loads(chrs)
    return Board(lst)

def get_score(board:Board):
    if board.is_ended():
        return {X:1, O:-1, BLANK:0}[board.winner]
    return 0

class Node:
    def __init__(self, board:Board):
        self.board = board
        self.score = get_score(board)
        self.children = []

def main():
    with sqlite3.connect("datas/evals.db") as conn:
        games = {}
        waiting = deque([Board()])
        
        while bool(waiting):
            board = waiting.popleft()
            
            if board in games:
                continue
            
            games[board] = Node(board)

            for move in board.get_legal_move():
                new_board = Board(board)
                current_player = board.get_current_player()
                new_board[move] = current_player
                waiting.append(new_board)
        print(len(games))