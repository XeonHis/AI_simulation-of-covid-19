import numpy as np
import sys

sys.path.append('../virl')
import virl
from matplotlib import pyplot as plt
import numpy as np


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


def random_agent():
    # get virl env
    env = virl.Epidemic()

    _states = []
    _rewards = []
    _actions = []
    done = False

    # reset the env
    s = env.reset()
    # add initial state into states
    _states.append(s)
    while not done:
        # set random action (0~3)
        random_action = int(np.random.random() * 4)
        s, r, done, i = env.step(action=random_action)
        _actions.append(random_action)
        _states.append(s)
        _rewards.append(r)
    return _actions, _states, _rewards


def return_locals_rpc_decorator(fun):
    def decorated_fun(*args, **kw):
        code = fun.func_code
        names = list(code.co_varnames[:code.co_argcount])
        args_in_list = list(args)
        print(names)

    return decorated_fun


@return_locals_rpc_decorator
def modified_random_agent():
    pass


print(modified_random_agent(random_agent))
# random_actions, random_states, random_rewards = random_agent()
# plot_figure(random_actions, random_states, random_rewards)
