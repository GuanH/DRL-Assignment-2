# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
from Game2048Wrapper import Game2048Wrapper


def softmax(x):
    x = np.exp(x - np.max(x))
    return x / np.sum(x)


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)),
                         mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j + 1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i + 1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def sim_afterstate(self, action):
        board = self.board.copy()
        score = self.score
        if action == 0:
            self.move_up()
        elif action == 1:
            self.move_down()
        elif action == 2:
            self.move_left()
        elif action == 3:
            self.move_right()
        ret_board = self.board.copy()
        ret_score = self.score
        self.board = board
        self.score = score
        return ret_board, ret_score

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1,
                                     1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(
            new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(
            new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)


# env = Game2048Env()
env = Game2048Wrapper()
env.reset()
"""
(0,0) (0,1) (0,2) (0,3)
(1,0) (1,1) (1,2) (1,3)
(2,0) (2,1) (2,2) (2,3)
(3,0) (3,1) (3,2) (3,3)
"""
# patterns = [[(0, 1), (1, 1), (2, 1), (3, 1)],
#             [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)],
#             [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)],
#             [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]]
# patterns = [
#     [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)],
#     [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (3, 0)],
#     [(0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2)],]

patterns = [[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
            [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)],
            [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)],
            [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)]]


def rot_flip_pattern(p, t):
    tem = copy.deepcopy(p)
    if t >= 4:
        for i in range(len(tem)):
            tem[i] = (tem[i][0], 3 - tem[i][1])
        t %= 4
    for i in range(t):
        for j in range(len(tem)):
            tem[j] = (tem[j][1], 3 - tem[j][0])
    return tem


def gen_sym(pattern):
    st = set()
    ret = []
    for i in range(8):
        x = rot_flip_pattern(pattern, i)
        x.sort()
        y = tuple(x)
        if y not in st:
            st.add(y)
            ret.append(x)
    return ret


sym_patterns = []
for i, p in enumerate(patterns):
    sp = gen_sym(p)
    for x in sp:
        sym_patterns.append((i, x))

with open('v_table.pkl', 'rb') as file:
    v_table = pickle.load(file)
def index(board, pattern):
    t = []
    for x, y in pattern:
        t.append((board[x, y]))
    return tuple(t)


def value(board):
    v = 0
    for i, p in sym_patterns:
        v += v_table[i][index(board, p)]
    return v


def select_value(a):
    score = env.get_score()
    board, new_score = env.sim_afterstate(a)
    return value(board) + new_score - score


def get_action(state, score):
    best_action = 0
    best_avg_score = 0
    score = int(score)
    env.set_board(state)
    env.set_score(score)
    moves = [a for a in range(4) if env.is_move_legal(a)]
    # return int(moves[np.argmax([select_value(a) for a in moves])])

    for i in moves:
        scores = []
        for _ in range(50):
            env.set_board(state)
            env.set_score(score)
            done = env.step(i)
            for _ in range(20):
                # while not done and env.get_score() < score * 2.5:
                legal_moves = [a for a in range(4) if env.is_move_legal(a)]
                if not legal_moves:
                    done = True
                    break
                action = int(legal_moves[np.argmax([select_value(a) for a in legal_moves])])
                done = env.step(action)
            scores.append(env.get_score() * 0.2 if done else value(env.get_board()))
            # scores.append(env.get_score())
        avg_score = np.mean(scores)
        if avg_score > best_avg_score:
            best_action = i
            best_avg_score = avg_score
    return best_action
