# from student_agent import Game2048Env
from Game2048Wrapper import Game2048Wrapper
import copy
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import pickle
EPOCH = 2000000

def softmax(x):
    x = np.exp(x - np.max(x))
    return x / np.sum(x)

env = Game2048Wrapper()
"""
(0,0) (0,1) (0,2) (0,3)
(1,0) (1,1) (1,2) (1,3)
(2,0) (2,1) (2,2) (2,3)
(3,0) (3,1) (3,2) (3,3)
"""
# patterns = [[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
#             [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)],
#             [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)],
#             [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)],
#             [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)],
#             [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 1)]]

patterns = [[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
            [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)],
            [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)],
            [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)]]

# patterns = [[(0, 1), (1, 1), (2, 1), (3, 1)],
#             [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)],
#             [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)],
#             [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]]
# patterns = [
#     [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)],
#     [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (3, 0)],
#     [(0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2)],]


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
v_table = [defaultdict(float) for _ in patterns]

with open('/tmp/v_table_c.pkl', 'rb') as file:
    v_table = pickle.load(file)
alpha = 0.0025


def index(board, pattern):
    t = []
    for x, y in pattern:
        t.append((board[x, y]))
    return tuple(t)


def value(board):
    v = 0
    for i, p in sym_patterns:
        id = index(board, p)
        v += v_table[i][id] if id in v_table[i] else 0
    return v


def select_value(a):
    score = env.get_score()
    board, new_score = env.sim_afterstate(a)
    return value(board) + new_score - score


def update(board, delta):
    for i, p in sym_patterns:
        v_table[i][index(board, p)] += delta


score_hist = []


def prepare_init(size, target_score, init_from_board=None, init_from_score=None):
    init_board = []
    init_score = []
    tqdm.write(f'Preparing init board, target score : {target_score}')
    for _ in tqdm(range(size)):
        while True:
            env.reset()
            if init_from_board:
                id = np.random.randint(0, len(init_from_board))
                env.set_board(init_from_board[id])
                env.set_score(init_from_score[id])

            done = False
            while not done and env.get_score() < target_score:
                legal_moves = [a for a in range(4) if env.is_move_legal(a)]
                if not legal_moves:
                    break
                action = int(legal_moves[np.argmax([select_value(a) for a in legal_moves])])
                done = env.step(action)

            if not done and env.get_score() >= target_score:
                init_board.append(env.get_board())
                init_score.append(env.get_score())
                break

    return init_board, init_score


target_score = 75000
ema_score = 37000

def reset_init():
    tem_board, tem_score = prepare_init(100, 10000)
    t = max(ema_score * 0.9 / 10000, 1)
    for i in range(2, int(t)):
        tem_board, tem_score = prepare_init(100, i * 10000, tem_board, tem_score)

    init_board, init_score = prepare_init(100, t * 10000, tem_board, tem_score)
    return init_board, init_score

init_board = []
init_score = []
init_board, init_score = reset_init()

for epoch in tqdm(range(EPOCH)):
    env.reset()
    rid = np.random.randint(0, 100)
    env.set_board(init_board[rid])
    env.set_score(init_score[rid])
    done = False
    step = 0
    while not done:
        step += 1
        after_score = env.get_score()
        after_state = env.get_board()
        env.add_random_tile()
        legal_moves = [a for a in range(4) if env.is_move_legal(a)]
        if not legal_moves:
            break
        action = int(legal_moves[np.argmax([select_value(a) for a in legal_moves])])

        done = env.afterstate(action)
        next_after_score = env.get_score()
        next_after_state = env.get_board()
        delta = alpha * ((next_after_score - after_score) + value(next_after_state) - value(after_state))
        update(after_state, delta)

    score_hist.append(env.get_score())

    if epoch % 100 == 99:
        avg_score = np.mean(score_hist)
        # if epoch > 50000:
        #     ema_score = 0.92 * ema_score + 0.08 * avg_score
        print(f'epoch : {epoch + 1:>6} score : {np.mean(score_hist):.4f} ({np.min(score_hist)}/{np.max(score_hist)})')
        score_hist = []

    if epoch % 10000 == 9999:
        #for table in v_table:
        #    keys = []
        #    for k, v in table.items():
        #        if v < 0.1:
        #            keys.append(k)
        #    print(f'Num : {len(keys)}')
        #    for k in keys:
        #        table.pop(k)
        with open('/tmp/v_table_c.pkl', 'wb') as file:
            pickle.dump(v_table, file)

    if epoch % 400 == 399:
        init_board, init_score = reset_init()

with open('/tmp/v_table_c.pkl', 'wb') as file:
    pickle.dump(v_table, file)
