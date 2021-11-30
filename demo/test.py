import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

sys.path.append('../virl')
import virl

env = virl.Epidemic(problem_id=5)

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000
N_ACTIONS = 4
N_STATES = 4


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        ori_q_eval = self.eval_net(b_s)  # shape (batch, 1)
        q_eval = ori_q_eval.gather(1, b_a)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train():
    dqn = DQN()

    print('\nCollecting experience...')
    best_reward = -np.inf
    for i_episode in range(400):
        s = env.reset()
        ep_r = 0
        while True:
            a = dqn.choose_action(s)

            # take action
            s_, r, done, info = env.step(a)

            dqn.store_transition(s, a, r, s_)

            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    if ep_r >= best_reward:
                        torch.save(dqn, 'covid2.pth')
                        print('****NEW MODEL****')
                        best_reward = ep_r

                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r[0], 2))

            if done:
                break
            s = s_


def test():
    model = torch.load('covid2.pth')
    torch.no_grad()

    _s = env.reset()
    done = False
    total_reward = 0
    while not done:
        actions_value = model.eval_net.forward(torch.unsqueeze(torch.FloatTensor(_s), 0))
        action = torch.max(actions_value, 1)[1].data.numpy()
        print(action)
        _next_state, reward, done, _ = env.step(action)
        _s = _next_state
        total_reward += reward
    print(total_reward)


if __name__ == '__main__':
    train()
    # test()