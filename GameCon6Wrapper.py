import ctypes
import numpy as np

lib = ctypes.CDLL('./con.so')
class CCon6(ctypes.Structure):
    _fields_ = [("board", (ctypes.c_int * 19) * 19),
                ("turn", ctypes.c_int)]

lib.create_env.restype = ctypes.POINTER(CCon6)

class Connect6Wrapper:
    def __init__(self):
        self.game = lib.create_env()

    def check_win(self):
        return lib.check_win_env(self.game)

    def play(self, x, y):
        lib.play_env(self.game, x, y)

    def render(self):
        lib.render_env(self.game)

    def get_board(self):
        return np.array(self.game.contents.board)
