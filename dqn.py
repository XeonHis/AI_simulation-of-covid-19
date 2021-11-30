import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import random
import sys
from collections import namedtuple, deque
from tensorboardX import SummaryWriter

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
        self.loss_func = nn.SmoothL1Loss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, inputs):
        return self.fc(inputs)


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def pop(self):
        return self.memory.pop()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


memory_size = 500
epsilon = 0.1
update_time = 10
gamma = 0.9
b_size = 32
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
MAX_EPISODE = 1000
memory = ReplayMemory(memory_size)


def run_dqn(_env, _approximator, _approximator_target):
    writer = SummaryWriter()
    best_reward = -np.inf
    learn_step = 0
    for i_episode in range(MAX_EPISODE):
        state = _env.reset()
        step = 0
        episode_reward = 0
        done = False
        actions = []
        rewards = []
        states = []
        while not done:
            if np.random.uniform() <= epsilon:
                action = random.randrange(0, 4)
            else:
                out = _approximator(torch.Tensor(state))
                action = torch.argmax(out).data.item()

            actions.append(action)
            next_state, reward, done, _ = _env.step(action)
            state = next_state
            episode_reward += reward

            memory.push(state, action, next_state, reward)

            if len(memory) >= memory_size:  # Start to learn
                if learn_step % update_time == 0:
                    _approximator_target.load_state_dict(_approximator.state_dict())
                learn_step += 1

                transitions = memory.sample(memory_size)
                batch = Transition(*zip(*transitions))

                b_s = torch.Tensor(np.array(batch.state))
                b_a = torch.Tensor(np.array(batch.action)).unsqueeze(1).long()
                b_s_ = torch.Tensor(np.array(batch.next_state))
                b_r = torch.Tensor(np.array(batch.reward)).unsqueeze(1)
                # print(b_s.shape, b_a.shape, b_s_.shape, b_r.shape)

                ori_q = _approximator(b_s)
                q = ori_q.gather(1, b_a)
                tq = b_r + gamma * _approximator_target(b_s_).detach().max(1)[0].unsqueeze(1)
                loss = _approximator.loss_func(q, tq)
                _approximator.opt.zero_grad()
                loss.backward()
                _approximator.opt.step()
                writer.add_scalar('loss', np.abs(torch.mean(tq - q.detach().numpy())), learn_step)

            if done:
                print('{}/{} Episode Reward={}'.format(i_episode, MAX_EPISODE, episode_reward))
                if episode_reward >= best_reward:
                    torch.save(_approximator, 'test_model/covid' + str(i_episode) + '.pth')
                    f = open('actions.txt', 'a+')
                    f.write(str(i_episode) + ':' + str(actions) + str(episode_reward) + '\n')
                    f.close()
                    print('****NEW MODEL****')
                    best_reward = episode_reward
                break
            step += 1


def test_dqn(_env):
    model = torch.load('test_model/covid60.pth').eval()
    torch.no_grad()

    _s = _env.reset()
    done = False
    total_reward = 0
    while not done:
        out = model(torch.Tensor(_s))
        action = torch.argmax(out).data.item()
        print(action)
        _next_state, reward, done, _ = _env.step(action)
        _s = _next_state
        total_reward += reward
    print(total_reward)


net = DeepQNetwork()
net2 = DeepQNetwork()
env = virl.Epidemic()

run_dqn(env, net, net2)

# test_dqn(env)
