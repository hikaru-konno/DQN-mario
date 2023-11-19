import numpy as np
import copy
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

class QNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=16):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_action)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        h = F.elu(self.fc3(h))
        y = F.elu(self.fc4(h))
        return y

class ReplayBuffer:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque([], maxlen = memory_size)

    def append(self, transition):
        #メモリに遷移を追加
        self.memory.append(transition)

    def sample(self, batch_size):
        #メモリからランダムにバッチサイズ分のインデックスを抽出
        batch_indexes = np.random.randint(0, len(self.memory), size=batch_size)
        states      = np.array([self.memory[index]['state'] for index in batch_indexes])
        next_states = np.array([self.memory[index]['next_state'] for index in batch_indexes])
        rewards     = np.array([self.memory[index]['reward'] for index in batch_indexes])
        actions     = np.array([self.memory[index]['action'] for index in batch_indexes])
        dones   = np.array([self.memory[index]['done'] for index in batch_indexes])
        return {'states': states, 'next_states': next_states, 'rewards': rewards, 'actions': actions, 'dones': dones}

class DqnAgent:
    def __init__(self, num_state, num_action, gamma=0.99, lr=0.001, batch_size=32, memory_size=50000):
        #状態の次元数
        self.num_state = num_state
        #行動の種類数
        self.num_action = num_action
        #割引率
        self.gamma = gamma 
        #バッチサイズ
        self.batch_size = batch_size  
        #Qネットワークの定義
        self.qnet = QNetwork(num_state, num_action)
        #ターゲットQネットワークの定義（Qネットワークと同じ構造で初期化）
        self.target_qnet = copy.deepcopy(self.qnet) 
        #最適化手法の定義（Adam）
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        #リプレイバッファの定義
        self.replay_buffer = ReplayBuffer(memory_size)
        #デバイスの設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qnet.to(self.device)
        self.target_qnet.to(self.device)

    def update_q(self):
        #リプレイバッファからバッチサイズ分のデータをサンプリング
        batch = self.replay_buffer.sample(self.batch_size)
        #Qネットワークに状態を入力してQ値を計算
        q = self.qnet(torch.tensor(batch["states"], dtype=torch.float).to(self.device))
        #Q値(ある状態である行動を取ったときに得られる報酬の期待値)のコピーを作成
        targetq = copy.deepcopy(q.data.cpu().numpy())
        #ターゲットQネットワークに次の状態を入力して最大Q値を計算
        maxq = torch.max(self.target_qnet(torch.tensor(batch["next_states"],dtype=torch.float).to(self.device)), dim=1).values

        for i in range(self.batch_size):
            #Q学習の更新式に従ってターゲットQ値を更新
            targetq[i, batch["actions"][i]] = batch["rewards"][i] + self.gamma * maxq[i] * (not batch["dones"][i])
        #勾配を初期化
        self.optimizer.zero_grad()
        #Qネットワークの出力とターゲットQ値の二乗誤差を計算
        loss = nn.MSELoss()(q, torch.tensor(targetq).to(self.device))
        #誤差逆伝播
        loss.backward()
        #パラメータの更新
        self.optimizer.step()
        #ターゲットQネットワークをQネットワークと同じにする
        self.target_qnet = copy.deepcopy(self.qnet)

    def get_greedy_action(self, state):
        #状態をテンソルに変換
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state).to(self.device)
        #Q値を計算
        action = torch.argmax(self.qnet(state_tensor).data).item()
        #Q値が最大の行動を返す
        return action

    def save_qnet(self):
        #モデルの保存
        with open('model.pickle', mode='wb') as f:
            pickle.dump(self.qnet, f, protocol=2)

    def load_qnet(self):
        #モデルの読み込み
        with open('model.pickle', mode='rb') as f:
            self.qnet = pickle.load(f)

    def get_action(self, state, episode):
        #ε-greedy法
        epsilon = 0.7 * (1/(episode+1))  
        if epsilon <= np.random.uniform(0,2):
            action = self.get_greedy_action(state)
        else:
            action = np.random.choice(self.num_action)
        return action