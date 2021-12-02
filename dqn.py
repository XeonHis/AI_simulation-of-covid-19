import torch
import torch.nn as nn
import numpy as np
import random
import sys
from collections import namedtuple, deque
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import torch.nn.functional as F

sys.path.append('virl')
import virl


class DeepQNetwork(nn.Module):
    def __init__(self, learning_rate):
        super(DeepQNetwork, self).__init__()
        self.lr = learning_rate
        self.fc = nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(24, 4)
        )
        self.loss_func = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inputs):
        inputs = F.normalize(inputs, dim=0)
        return self.fc(inputs)


class ReplayMemory:

    def __init__(self, _transition, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.inner_transition = _transition

    def push(self, *args):
        self.memory.append(self.inner_transition(*args))

    def pop(self):
        return self.memory.pop()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def plot_figure(_actions, _states, _rewards):
    # start a figure with 2 subplot
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    labels = ['s[0]: susceptibles', 's[1]: infectious', 's[2]: quarantined', 's[3]: recovereds']
    _states = np.array(_states)

    # plot state evolution on the left subplot
    for i in range(4):
        axes[0].plot(_states[:, i], label=labels[i])
    axes[0].set_title('State')
    axes[0].set_xlabel('Weeks since start of epidemic')
    axes[0].set_ylabel('State s(t)')
    axes[0].legend()

    # plot reward evolution on the right subplot
    axes[1].plot(_rewards)
    axes[1].set_title('Reward')
    axes[1].set_xlabel('Weeks since start of epidemic')
    axes[1].set_ylabel('Reward r(t)')

    axes[2].plot(_actions)
    axes[2].set_title('Action')
    axes[2].set_xlabel('Weeks since start of epidemic')
    axes[2].set_ylabel('Action a(t)')

    print('Total reward for this episode is ', np.sum(_rewards))
    plt.show()


def run_dqn(_env, _approximator, _approximator_target, _batch_size, _max_episode, _memory_size, _update_time, _gamma,
            _transition, _memory, _learning_rate):
    writer = SummaryWriter(
        'runs_model/model_episode={}_lr={}_bsize={}_utime={}'.format(_max_episode, _learning_rate, _batch_size,
                                                                     _update_time))
    best_reward = -np.inf
    learn_step = 0
    epsilon = 0.1
    epsilon_min = 0.01
    decay_factor = 0.99995
    best_actions = None
    best_rewards = None
    best_states = None
    for i_episode in range(_max_episode):
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

            next_state, reward, done, _ = _env.step(action)

            actions.append(action)
            rewards.append(reward)
            states.append(state)

            state = next_state
            episode_reward += reward

            _memory.push(state, action, next_state, reward)

            if len(_memory) >= _memory_size:  # Start to learn
                if learn_step % _update_time == 0:
                    _approximator_target.load_state_dict(_approximator.state_dict())
                learn_step += 1

                transitions = _memory.sample(_batch_size)
                batch = _transition(*zip(*transitions))

                b_s = torch.Tensor(np.array(batch.state))
                b_a = torch.Tensor(np.array(batch.action)).unsqueeze(1).long()
                b_s_ = torch.Tensor(np.array(batch.next_state))
                b_r = torch.Tensor(np.array(batch.reward)).unsqueeze(1)
                # print(b_s.shape, b_a.shape, b_s_.shape, b_r.shape)

                ori_q = _approximator(b_s)
                q = ori_q.gather(1, b_a)
                # temp = _approximator_target(b_s_).detach()
                # temp2 = temp.max(1)
                # temp3 = temp2[0]
                # temp4 = _gamma * temp3
                tq = b_r + _gamma * _approximator_target(b_s_).detach().max(1)[0].unsqueeze(1)
                loss = _approximator.loss_func(q, tq)
                _approximator.opt.zero_grad()
                loss.backward()
                _approximator.opt.step()

                writer.add_scalar('loss', loss.detach().numpy(), learn_step)

            if done:
                print('{}/{} Episode Reward={}'.format(i_episode, _max_episode, episode_reward))
                writer.add_scalar('runs_model/episode_reward', episode_reward, i_episode)
                if episode_reward >= best_reward:
                    torch.save(_approximator.state_dict(),
                               'model/test/lr={}_bsize={}_utime={}.pkl'.format(_learning_rate, _batch_size,
                                                                               _update_time))
                    f = open('actions.txt', 'w+')
                    f.write(str(i_episode) + ':' + str(actions) + str(episode_reward) + '\n')
                    f.close()
                    print('****NEW MODEL****')
                    best_reward = episode_reward
                    best_actions = actions.copy()
                    best_states = states.copy()
                    best_rewards = rewards.copy()
                break

            if epsilon > epsilon_min:
                epsilon *= decay_factor
            step += 1

        if i_episode == _max_episode - 1:
            torch.save(_approximator.state_dict(),
                       'model/test/lr={}_bsize={}_utime={}_last.pkl'.format(_learning_rate, _batch_size, _update_time))
            f = open('actions.txt', 'a+')
            f.write('**LAST** ' + str(i_episode) + ':' + str(actions) + str(episode_reward))
            f.close()
            print('****LAST MODEL****')
    plot_figure(best_actions, best_states, best_rewards)


def test_dqn(_env, _net, _path):
    # _path default: model/relu/lr=0.02_msize=2_last.pkl
    _net.load_state_dict(torch.load(_path))

    _s = _env.reset()
    done = False
    rewards = []
    states = []
    actions = []
    while not done:
        out = _net(torch.Tensor(_s))
        action = torch.argmax(out).data.item()
        _next_state, reward, done, _ = _env.step(action)
        rewards.append(reward)
        states.append(_s)
        actions.append(action)
        _s = _next_state
    plot_figure(actions, states, rewards)


def train(_env, _net, _net2, _batch_size, _max_episode, _memory_size, _update_time, _gamma, _transition, _memory,
          _learning_rate):
    for m in _net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    for m in _net2.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    run_dqn(_env, _net, _net2, _batch_size, _max_episode, _memory_size, _update_time, _gamma, _transition, _memory,
            _learning_rate)


def DQN_agent(env, train_mode, batch_size=32, learning_rate=0.02, max_episode=1000, memory_size=2000, update_time=100,
              gamma=0.9, path='model/test/lr=0.02_msize=32.pkl'):
    net = DeepQNetwork(learning_rate)
    net2 = DeepQNetwork(learning_rate)
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    memory = ReplayMemory(Transition, memory_size)
    if train_mode:
        train(env, net, net2, batch_size, max_episode, memory_size, update_time, gamma, Transition, memory,
              learning_rate)
    else:
        test_dqn(env, net, path)


if __name__ == '__main__':
    env = virl.Epidemic()
    DQN_agent(env, train_mode=False, batch_size=32, learning_rate=0.02, max_episode=1000, memory_size=15000,
              update_time=100)
