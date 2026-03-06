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

    def get_key(self):
        return self.board.get_key()
    
    def __hash__(self):
        return self.board.__hash__()
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.board == other.board
        
class Root(Node):
    pass

def get_id(board, cursor):
    board_key = board.get_key()
    
    cursor.execute("""
        SELECT id FROM games
        WHERE key = ?
    """, (board_key,))
    
    result = cursor.fetchone()
    return result[0] if result else None

def create_games_database(verbose = True):
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
                key STRING UNIQUE,
                player STRING,
                end INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                score FLOAT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                childId INTEGER,
                parentId INTEGER
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
                    key,
                    player,
                    end
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (*get_database_format_board(board), node.get_key(), {X: "x", O: "o"}[board.get_current_player()], int(board.is_ended())))
            
            cursor.execute("""
                INSERT INTO scores (
                    score
                )
                VALUES (?)
            """, (node.score,))

    for board, node in games.items():
        current_id = get_id(node, cursor)
            
        for child in node.children:
            cursor.execute("""
                INSERT INTO relations (
                    childId,
                    parentId
                )
                VALUES (?, ?)
            """, (get_id(child, cursor), current_id))

    conn.commit()
        
def is_full():
    n = 6046
    
    try:
        with sqlite3.connect("datas/evals.db") as conn:
            c = conn.cursor()
            c.execute("""
                SELECT COUNT(*) FROM games
            """)
            res = c.fetchone()[0]
    except:
        return False
    
    return res == n

def get_end_games(conn:sqlite3.Connection) -> list:
    """
    Renvoie la liste des identifiants des positions gagnantes ou perdantes 
    """
    c = conn.cursor()
    c.execute("""
        SELECT games.id FROM games
        WHERE end = 1
    """)
    
    return [el[0] for el in c.fetchall()]
    
def get_data_score(id:int, conn:sqlite3.Connection) -> float:
    """
    Renvoie le score de la partie correspondante
    """
    c = conn.cursor()
    c.execute("""
        SELECT score FROM scores
        WHERE id = ?
    """, (id,))
        
    return c.fetchone()[0]

def get_parents_id(id:int, conn:sqlite3.Connection) -> list:
    """
    Renvoie la liste des parents (précédant un coups) de la partie
    """
    c = conn.cursor()
    c.execute("""
        SELECT parentId FROM relations
        WHERE childId = ?
    """, (id,))
        
    return [el[0] for el in c.fetchall()]

def update_score(id:int, score:float, conn:sqlite3.Connection):
    c = conn.cursor()
    c.execute("""
        UPDATE scores
        SET score = ?
        WHERE id = ?
    """, (score, id))
    
def get_player(id:int, conn:sqlite3.Connection) -> int:
    c = conn.cursor()
    c.execute("""
        SELECT player FROM games
        WHERE id = ?
    """, (id,))
    
    return {"x":X, "o":O}[c.fetchone()[0]]

def value_assignation(f:float = 1):
    """
    Propage les scores des positions de fin vers les positions enfants
    Permet le calcul de la probabilité de victoire ou de défaite d'une partie
    """
    conn = sqlite3.connect("datas/evals.db")
    last_layer = get_end_games(conn)
    idx = 0

    while last_layer != []:
        print(last_layer)
        idx += 1
        print(f"Layer {idx} \r")
        scores = {}
        temp_current_layer = []
        for current_id in last_layer:
            current_player = get_player(current_id, conn)
            parents_ids = get_parents_id(current_id, conn)
            current_score = get_data_score(current_id, conn)
            
            for parent_id in parents_ids:
                if parent_id in scores:
                    n = scores[parent_id]
                    
                    if current_player == X:
                        n = max(n, scores[parent_id])
                    else:
                        n = min(n, scores[parent_id])
                    
                    scores[parent_id] = n
                else:
                    scores[parent_id] = current_score
            
            temp_current_layer += parents_ids
        print(scores)
        for id, score in scores.items():
            update_score(id, score * f, conn)
        
        last_layer = list(set(temp_current_layer.copy()))
    conn.commit()
    conn.close()

def test_evaluation():
    conn = sqlite3.connect("datas/evals.db")
    cursor = conn.cursor()
    g = Game()

    for last_move, last_player, moves, winner in g.play():
        print(last_move, last_player, moves, winner)
        id = get_id(g.board, cursor)
        evaluation = get_data_score(id, conn)
        print(evaluation)
    conn.close()

def main(verbose = True):
    if not is_full():
        create_games_database(verbose)
        value_assignation()
    test_evaluation()
