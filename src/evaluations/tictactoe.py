import sqlite3
import json

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
    

def main():
    with sqlite3.connect("datas/evals.db") as conn:
        games = {}
        waiting = [Board()]
        
        while waiting != []:
            board = waiting.pop()
            
            if board in games:
                continue
            
            games[board] = board