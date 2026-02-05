import sqlite3
import json
import time
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

def get_database_format_board(board:Board):
    DATABASE_VALS = {
        X: "x",
        O: "o",
        BLANK: "b",
    }
    res = []
    
    for row in board:
        res.extend([DATABASE_VALS[val] for val in row])
    
    return tuple(res)
        
class Node:
    def __init__(self, board:Board):
        self.board = board
        self.score = get_score(board)
        self.children = []
        self.parents = []
    
    def __hash__(self):
        return self.board.__hash__()
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.board == other.board
        
class Root(Node):
    pass

def main(verbose = True):
    idx = 0
    last = time.time()
    games = {}
    waiting = deque([Root(Board())])
    
    while bool(waiting):
        cur_node = waiting.popleft()
        
        if cur_node in games:
            continue
        
        board = cur_node.board
        games[board] = cur_node

        for move in board.get_legal_move():
            new_board = Board(board)
            current_player = board.get_current_player()
            new_board[move] = current_player
            new_node = Node(new_board)
            new_node.parents.append(cur_node)
            cur_node.children.append(new_node)
            waiting.append(new_node)
        if time.time() - last > 3:
            last = time.time()
            if verbose:
                print(f"\rPosition : {idx}, Parties : {len(games)}", end="")
        idx += 1
            
    with sqlite3.connect("datas/evals.db") as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topLeftSquare STRING,
                topMiddleSquare STRING,
                topRightSquare STRING,
                middleLeftSquare STRING,
                middleMiddleSquare STRING,
                middleRightSquare STRING,
                bottomLeftSquare STRING,
                bottomMiddleSquare STRING,
                bottomRightSquare STRING,
                hash STRING UNIQUE,
                player STRING
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                score FLOAT
            )
        """)

        for board, node in games.items():
            cursor.execute("""
                INSERT INTO games (
                    topLeftSquare,
                    topMiddleSquare,
                    topRightSquare,
                    middleLeftSquare,
                    middleMiddleSquare,
                    middleRightSquare,
                    bottomLeftSquare,
                    bottomMiddleSquare,
                    bottomRightSquare,
                    hash,
                    player
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (*get_database_format_board(board), node.__hash__(), {X: "x", O: "o"}[board.get_current_player()]))
            
            cursor.execute("""
                INSERT INTO scores (
                    score
                )
                VALUES (?)
            """, (node.score,))

        conn.commit()
