import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
from tqdm import tqdm
from GameCon6Wrapper import Connect6Wrapper

def softmax(x):
    x = np.exp(x - np.max(x))
    return x / np.sum(x)

env = Connect6Wrapper()
sim_env = Connect6Wrapper()

EPOCH = 5000000
device = torch.device("cuda")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # (1, 19, 19, 3)
        self.conx1 = nn.Sequential(nn.Conv2d(3, 16, 5), nn.SiLU(inplace=True))
        self.conx2 = nn.Sequential(nn.Conv2d(16, 32, 5), nn.SiLU(inplace=True))
        self.conx3 = nn.Sequential(nn.Conv2d(32, 32, 3), nn.SiLU(inplace=True))
        self.conx4 = nn.Sequential(nn.Conv2d(32, 32, 3), nn.SiLU(inplace=True))
        self.conx5 = nn.Sequential(nn.Conv2d(32, 16, 3), nn.SiLU(inplace=True))
        self.linear = nn.Sequential(
            nn.Linear(400, 1024), nn.SiLU(inplace=True),
            nn.Linear(1024, 1024), nn.SiLU(inplace=True),
            nn.Linear(1024, 400), nn.SiLU(inplace=True),
        )
        self.cony5 = nn.Sequential(nn.ConvTranspose2d(16, 32, 3), nn.SiLU(inplace=True))
        self.cony4 = nn.Sequential(nn.ConvTranspose2d(32, 32, 3), nn.SiLU(inplace=True))
        self.cony3 = nn.Sequential(nn.ConvTranspose2d(32, 32, 3), nn.SiLU(inplace=True))
        self.cony2 = nn.Sequential(nn.ConvTranspose2d(32, 16, 5), nn.SiLU(inplace=True))
        self.cony1 = nn.ConvTranspose2d(16, 2, 5)

    # (B, 3, 19, 19) -> (B, 19*19)
    def forward(self, x):
        B = x.shape[0]
        x1 = self.conx1(x)
        x2 = self.conx2(x1)
        x3 = self.conx3(x2)
        x4 = self.conx4(x3)
        x5 = self.conx5(x4)
        y5 = x5 + self.linear(x5.reshape((B, -1))).reshape((B, 16, 5, 5))
        y4 = x4 + self.cony5(y5)
        y3 = x3 + self.cony4(y4)
        y2 = x2 + self.cony3(y3)
        y1 = x1 + self.cony2(y2)
        # (B, 2, 19, 19)
        y = self.cony1(y1).reshape((B, 2, 361))
        return y

class TargetAgentNN:
    def __init__(self, id):
        self.id = id
        self.model = Model().to(device)
        self.reload()

    def reload(self):
        try:
            self.model.load_state_dict(torch.load(f'./Con6Models/Con6Model_{self.id}.pt'))
        except:
            print(f'Failed to load target model ./Con6Models/Con6Model_{self.id}.pt')
            pass

    def play(self, board, target_eps=0):
        mask = torch.zeros(19 * 19)
        mask[board.reshape(19 * 19) != 0] = float('-inf')
        encode_board, p1, p2 = policy(board, self.model)
        if np.random.random() < target_eps:
            action1 = np.random.choice(len(p1), p=F.softmax(mask, dim=-1).numpy())
        else:
            action1 = int(np.argmax((p1.detach().cpu() + mask).numpy()))

        mask[action1] = float('-inf')
        if np.random.random() < target_eps:
            action2 = np.random.choice(len(p2), p=F.softmax(mask, dim=-1).numpy())
        else:
            action2 = int(np.argmax((p2.detach().cpu() + mask).numpy()))

        return action1, action2

class TargetAgentRuleBased1:
    def __init__(self):
        pass

    def reload(self):
        pass

    def play(self, board, target_eps=0):
        # play randomly near played stones
        p = (board == 0).astype(np.float32)
        for i in range(19):
            for j in range(19):
                if board[i,j] == 2:
                    p[i, min(j + 1, 18)] *= 100
                    p[i, max(j - 1, 0)] *= 100
                    p[min(i + 1, 18), j] *= 100
                    p[max(i - 1, 0), j] *= 100
                    p[min(i + 1, 18), min(j + 1,18)] *= 100
                    p[min(i + 1, 18), max(j - 1,0)] *= 100
                    p[max(i - 1, 0), min(j + 1,18)] *= 100
                    p[max(i - 1, 0), max(j - 1,0)] *= 100
        p = np.clip(p.reshape(-1), 0, 100)
        actions = np.random.choice(19 * 19, size=2, p=p / p.sum())
        p = (board == 0).astype(np.float32).reshape(-1)
        if np.random.random() < target_eps:
            actions[0] = np.random.choice(19 * 19, p=p / p.sum())
            p[actions[0]] = 0

        if np.random.random() < target_eps:
            actions[1] = np.random.choice(19 * 19, p=p / p.sum())
        return int(actions[0]), int(actions[1])

class TargetAgentRuleBased2:
    def __init__(self):
        pass

    def reload(self):
        pass

    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]

    def get_cost(self, board, x, y):
        best_cost = 7
        a1 = 0
        a2 = 0
        for i in np.random.permutation(8):
            cost = 6
            suc = True
            ca1 = -1
            ca2 = -1
            for j in range(6):
                cx, cy = x + TargetAgentRuleBased2.dx[i] * j, y + TargetAgentRuleBased2.dy[i] * j
                if cx < 0 or cx >= 19 or cy < 0 or cy >= 19:
                    suc = False
                    break
                cur = board[cx, cy]
                if cur == 2:
                    suc = False
                    break
                elif cur == 1:
                    cost -= 1
                else:
                    if ca1 == -1:
                        ca1 = cx * 19 + cy
                    elif ca2 == -1:
                        ca2 = cx * 19 + cy
            if suc and cost < best_cost:
                a1, a2 = ca1, ca2
                best_cost = cost
        return best_cost, a1, a2

    def play(self, board, target_eps=0):
        # Greedy
        action1 = 0
        action2 = 0
        best_cost = 7
        for i in np.random.permutation(19):
            for j in np.random.permutation(19):
                if board[i,j] != 2:
                    cost, a1, a2 = self.get_cost(board, i, j)
                    if board[i,j] == 1:
                        cost -= 0.5
                    if cost < best_cost:
                        action1 = a1
                        action2 = a2
                        best_cost = cost

        p = (board == 0).astype(np.float32).reshape(-1)

        if np.random.random() < target_eps:
            action1 = np.random.choice(19 * 19, p=p / p.sum())
            p[action] = 0

        if action2 == -1 or np.random.random() < target_eps:
            action2 = np.random.choice(19 * 19, p=p / p.sum())


        return int(action1), int(action2)


TRAIN_ID = []
NUM_MODELS = 0
if len(sys.argv) >= 2:
    try:
        TRAIN_ID = list(map(int, sys.argv[1].split(',')))
        NUM_MODELS = len(TRAIN_ID)
        print(f'Train on : {TRAIN_ID}')
    except:
        exit()

train_models = [Model().to(device) for _ in range(NUM_MODELS)]
target_models = [TargetAgentNN(i) for i in range(20)] + [TargetAgentRuleBased1(), TargetAgentRuleBased2()]
NUM_TARGET = len(target_models)

for i, model in enumerate(train_models):
    try:
        model.load_state_dict(torch.load(f'./Con6Models/Con6Model_{TRAIN_ID[i]}.pt'))
    except:
        print(f'Failed to load train model ./Con6Models/Con6Model_{TRAIN_ID[i]}.pt')
        pass

opts = [torch.optim.AdamW(model.parameters()) for model in train_models]


def get_encode_board(board):
    x = np.zeros((1, 3, 19, 19), dtype=np.float32)
    for i in range(19):
        for j in range(19):
            x[0,board[i,j],i,j] = 1
    return x


def policy(board, model):
    with torch.no_grad():
        x = get_encode_board(board)
        p = model(torch.tensor(x, device=device))[0]
        return x[0], p[0].reshape(19 * 19), p[1].reshape(19 * 19)


loss_hist = [0]
eps = 0.1
eps_rate = (0.01 / eps)**(1.0 / EPOCH)
win_rate = np.zeros(NUM_MODELS, dtype=np.float32)
win_cnt = np.zeros(2 * NUM_TARGET)
cnt = np.zeros(2 * NUM_TARGET)

model_id = 0
opts[model_id].zero_grad()
target_turn_id = 0


for epoch in tqdm(range(EPOCH)):
    env.reset()
    done = False
    X1 = []
    A10 = []
    A11 = []

    X2 = []
    A20 = []
    A21 = []
    target_turn_id = np.argmin(win_cnt)
    if np.random.random() < 0.3:
        target_turn_id = np.random.randint(0, 2 * NUM_TARGET)
    target_id = target_turn_id // 2
    model_turn = 2 - (target_turn_id % 2)
    final_turn = 0
    first = True
    x = win_cnt[target_turn_id] / cnt[target_turn_id] if cnt[target_turn_id] != 0 else 1
    eps = 0.01 * ((1 - x) ** 4)
    target_eps = (1 - min(x / 0.3, 1)) ** 2
    target_eps = 0
    while not done:
        board = env.get_board()
        turn = env.get_turn()

        if turn == 2:
            board = (3 - board) % 3

        final_turn = env.get_turn()
        if turn == model_turn:
            encode_board, p1, p2 = policy(board, train_models[model_id])
            mask = torch.zeros(19 * 19)
            mask[board.reshape(19 * 19) != 0] = float('-inf')
            temp = 1 if win_cnt[target_turn_id] / max(cnt[target_turn_id], 1) > 0.35 else 0.95
            if np.random.random() < eps:
                action = np.random.choice(len(p1), p=F.softmax(mask, dim=-1).numpy())
            else:
                p1 = F.softmax(p1.detach().cpu() * temp + mask, dim=-1).numpy()
                action = np.random.choice(len(p1), p=p1)
            X1.append(encode_board)
            A10.append(action)
            env.play(action // 19, action % 19)

            mask[action] = float('-inf')
            if np.random.random() < eps:
                action = np.random.choice(len(p2), p=F.softmax(mask, dim=-1).numpy())
            else:
                p2 = F.softmax(p2.detach().cpu() * temp + mask, dim=-1).numpy()
                action = np.random.choice(len(p2), p=p2)
            A11.append(action)
            if not first:
                env.play(action // 19, action % 19)
        else:
            action1, action2 = target_models[target_id].play(board, target_eps)
            X2.append(get_encode_board(board)[0])
            A20.append(action1)
            A21.append(action2)
            env.play(action1 // 19, action1 % 19)
            if not first:
                env.play(action2 // 19, action2 % 19)


        first = False

        done = env.check_win()
        if done:
            break
    cnt[target_turn_id] += 1
    if final_turn == model_turn:
        win_cnt[target_turn_id] += 1
        X = np.array(X1)
        A0 = torch.tensor(np.array(A10), device=device, dtype=torch.int64)
        A1 = torch.tensor(np.array(A11), device=device, dtype=torch.int64)
        r = torch.tensor([0.99**i for i in reversed(range(len(X)))], device=device)
        p = train_models[model_id](torch.tensor(X, device=device)).reshape((-1, 2, 361))
        logp = -F.cross_entropy(p[:,0], A0, reduction='none') - F.cross_entropy(p[:,1], A1, reduction='none')
        loss = (-logp * r).sum()
        loss_hist.append(loss.item())
        loss.backward()
        if win_cnt.sum() % 64 == 63:
            nn.utils.clip_grad_norm_(train_models[model_id].parameters(), 1.0)
            opts[model_id].step()
            opts[model_id].zero_grad()


    eps *= eps_rate
    if epoch % 500 == 499:
        win_rate[model_id] = win_cnt.sum() / cnt.sum()
        tqdm.write(f'epoch : {epoch + 1} loss : {np.mean(loss_hist):.8f} model{TRAIN_ID[model_id]} win rate : {win_rate[model_id]:.4f}')
        loss_hist = [0]
        if cnt.sum() >= 500:
            id = np.argmin(np.clip(win_rate, 0, 0.5) + 0.001 * np.random.random(NUM_MODELS))
            if id != model_id:
                nn.utils.clip_grad_norm_(train_models[model_id].parameters(), 1.0)
                opts[model_id].step()
                opts[model_id].zero_grad()
                model_id = id
                opts[model_id].zero_grad()
                win_cnt = np.zeros(2 * NUM_TARGET)
                cnt = np.zeros(2 * NUM_TARGET)
                target_turn_id = 0

    if epoch % 10000 == 9999:
        for i, model in enumerate(train_models):
            torch.save(model.state_dict(), f'./Con6Models/Con6Model_{TRAIN_ID[i]}.pt')

        for model in target_models:
            model.reload()

        win_rate = np.zeros(NUM_MODELS, dtype=np.float32)
        win_cnt = np.zeros(2 * NUM_TARGET)
        cnt = np.zeros(2 * NUM_TARGET)


for i, model in enumerate(train_models):
    torch.save(model.state_dict(), f'./Con6Models/Con6Model_{TRAIN_ID[i]}.pt')
