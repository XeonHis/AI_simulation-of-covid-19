import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import random
import sys

sys.path.append('virl')
import virl


# np.random.seed(2)


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 4)
        )
        self.mse = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, inputs):
        return self.fc(inputs)


memory_size = 50
epsilon = 0.1
update_time = 50
gama = 0.9
b_size = 32
memory = np.zeros((memory_size, 10))  # S(4)   A(1)   S_(4)   R(1)
MAX_EPISODE = 2000


def run_deepQ(_env, _approximator, _approximator_target):
    memory_count = 0
    learn_time = 0
    best_reward = -np.inf

    for i_episode in range(MAX_EPISODE):
        observation = _env.reset()
        step = 0
        episode_reward = 0
        while True:
            if np.random.rand() <= epsilon:
                action = random.randrange(4)
            else:
                out = _approximator(torch.Tensor(observation)).detach()
                action = torch.argmax(out).data.item()

            observation_, reward, done, _ = _env.step(action)
            episode_reward += reward

            observation = observation_

            idx = memory_count % memory_size
            memory[idx][0:4] = observation
            memory[idx][4:5] = action
            memory[idx][5:9] = observation_
            memory[idx][9:10] = reward
            memory_count += 1

            if memory_count >= memory_size:  # Start to learn
                learn_time += 1  # Learn once
                if learn_time % update_time == 0:  # Sync two nets
                    _approximator_target.load_state_dict(_approximator.state_dict())
                else:
                    rdp = random.randint(0, memory_size - b_size - 1)
                    b_s = torch.Tensor(memory[rdp:rdp + b_size, 0:4])
                    b_a = torch.Tensor(memory[rdp:rdp + b_size, 4:5]).long()
                    b_s_ = torch.Tensor(memory[rdp:rdp + b_size, 5:9])
                    b_r = torch.Tensor(memory[rdp:rdp + b_size, 9:10])

                    ori_q = _approximator(b_s)
                    q = ori_q.gather(1, b_a)
                    q_next = _approximator_target(b_s_).detach().max(1)[0].reshape(b_size, 1)
                    tq = b_r + gama * q_next
                    loss = _approximator.mse(q, tq)
                    _approximator.opt.zero_grad()
                    loss.backward()
                    _approximator.opt.step()

            step += 1
            if done:
                print('{}/{} Episode Reward={}'.format(i_episode, MAX_EPISODE, episode_reward))
                if episode_reward > best_reward:
                    torch.save({'model': _approximator.state_dict()}, 'covid.pth')
                    print('****NEW MODEL****')
                    best_reward = episode_reward
                break


def test_dqn(_env, _net):
    test_model = _net
    state_dict = torch.load('covid.pth')
    test_model.load_state_dict(state_dict['model'])

    observation = _env.reset()
    done = False
    total_reward = 0
    while not done:
        out = test_model(torch.Tensor(observation)).detach()
        action = torch.argmax(out).data.item()
        print(action)
        observation_, reward, done, _ = _env.step(action)
        total_reward += reward
    print(total_reward)


net = DeepQNetwork()
net2 = DeepQNetwork()
env = virl.Epidemic()

run_deepQ(env, net, net2)

# test_dqn(env, net)
