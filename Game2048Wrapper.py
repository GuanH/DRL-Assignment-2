import ctypes
import numpy as np

lib = ctypes.CDLL('./2048.so')


class CGame2048(ctypes.Structure):
    _fields_ = [("board", (ctypes.c_int*4)*4),
                ("score", ctypes.c_int),
                ("sim_board", (ctypes.c_int*4)*4),
                ("sim_score", ctypes.c_int)]


lib.create_game_2048.restype = ctypes.POINTER(CGame2048)


class Game2048Wrapper:
    def __init__(self):
        self.game = lib.create_game_2048()

    def reset(self):
        lib.reset_2048(self.game)
        self.add_random_tile()

    def step(self, action):
        done = lib.step_2048(self.game, action)
        return done

    def afterstate(self, action):
        done = lib.afterstate_2048(self.game, action)
        return done

    def add_random_tile(self):
        lib.add_random_tile_2048(self.game)

    def is_move_legal(self, action):
        return lib.is_move_legal_2048(self.game, action)

    def render(self):
        lib.render_2048(self.game)

    def get_score(self):
        return self.game.contents.score

    def get_board(self):
        return np.array(self.game.contents.board)

    def sim_step(self, action):
        lib.sim_step_2048(self.game, action)
        return np.array(self.game.contents.sim_board), self.game.contents.sim_score

    def sim_afterstate(self, action):
        lib.sim_afterstate_2048(self.game, action)
        return np.array(self.game.contents.sim_board), self.game.contents.sim_score
