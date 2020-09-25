import random
import math
import torch
import torch.distributions.beta as beta_dist


gittins_idx = None


def sample_beta_dist(alpha, beta):
    return beta_dist.Beta(alpha, beta).sample()


def load_gittins(file_name):
    result = []
    with open(file_name, 'r') as f:
        for line in f:
            r_line = []
            for val in line.split():
                r_line.append(float(val))
            result.append(r_line)
    return torch.tensor(result)


def ucb(machine):
    s = torch.tensor([0.0, 0.0])
    n = torch.tensor([1.0, 1.0])
    time = machine.time_limit
    actions = [0] * time
    rewards = [0] * time
    probs = [0] * time

    for t in range(time):
        machine.set_time(t)
        # ucb action selection
        val = s / n + torch.sqrt(
            2 * torch.log(torch.tensor([t+1]).float()) / n)
        action = val.argmax().item()

        reward = machine.sample_reward(action)

        # update
        s[action] += reward
        n[action] += 1

        # recording
        actions[t] = action
        rewards[t] = reward
        probs[t] = machine.check_optimality(action)

    return rewards, probs


def thompson(machine):
    s = torch.tensor([1.0, 1.0])
    n = torch.tensor([1.0, 1.0])
    time = machine.time_limit
    actions = [0] * time
    rewards = [0] * time
    probs = [0] * time

    for t in range(time):
        machine.set_time(t)

        # thompson sampling action selection
        val = torch.tensor([sample_beta_dist(s[0], n[0]),
                            sample_beta_dist(s[1], n[1])])
        action = val.argmax().item()

        reward = machine.sample_reward(action)

        # update
        s[action] += reward
        n[action] += 1 - reward

        # recording
        actions[t] = action
        rewards[t] = reward
        probs[t] = machine.check_optimality(action)

    return rewards, probs


def epsilon_greedy(machine, epsilon=0.05):
    s = torch.tensor([0.0, 0.0])
    n = torch.tensor([1.0, 1.0])
    time = machine.time_limit
    actions = [0] * time
    rewards = [0] * time
    probs = [0] * time

    for t in range(time):
        machine.set_time(t)

        # e-greedy action selection
        exploring = (random.random() < epsilon)
        if exploring:
            action = 1 if random.random() < 0.5 else 0
        else:
            action = (s / n).argmax().item()

        reward = machine.sample_reward(action)

        # update
        s[action] += reward
        n[action] += 1

        # recording
        actions[t] = action
        rewards[t] = reward
        probs[t] = machine.check_optimality(action)

    return rewards, probs


def exp3(machine):
    w = torch.tensor([1.0, 1.0])
    gamma = 0.1
    time = machine.time_limit
    actions = [0] * time
    rewards = [0] * time
    probs = [0] * time

    for t in range(time):
        machine.set_time(t)

        # exp3 action selection
        sum_w = w.sum()
        dist = (1.0 - gamma) * w / sum_w + gamma / w.size(0)
        choice = random.random()
        action = 0 if dist[0] >= choice else 1

        reward = machine.sample_reward(action)

        # update
        e_r = 1.0 * reward / dist[action].item()
        w[action] *= math.exp(e_r * gamma / w.size(0))

        # recording
        actions[t] = action
        rewards[t] = reward
        probs[t] = machine.check_optimality(action)

    return rewards, probs


def gittins(machine):
    s = torch.tensor([0.0, 0.0])
    n = torch.tensor([0.0, 0.0])
    time = machine.time_limit
    actions = [0] * time
    rewards = [0] * time
    probs = [0] * time
    global gittins_idx
    if gittins_idx is None:
        gittins_idx = load_gittins('gittins.txt')

    for t in range(time):
        machine.set_time(t)
        # gittins action selection
        val = torch.zeros(2)
        val[0] = gittins_idx[int(n[0]+1), int(s[0]+1)]
        val[1] = gittins_idx[int(n[1]+1), int(s[1]+1)]

        action = val.argmax().item()

        reward = machine.sample_reward(action)

        # update
        s[action] += reward
        n[action] += 1 - reward

        # recording
        actions[t] = action
        rewards[t] = reward
        probs[t] = machine.check_optimality(action)

    return rewards, probs


def observational(machine):
    s = torch.tensor([0.0, 0.0])
    n = torch.tensor([1.0, 1.0])
    time = machine.time_limit
    actions = [0] * time
    rewards = [0] * time
    probs = [0] * time

    for t in range(time):
        machine.set_time(t)

        # observational action selection
        action = machine.get_hidden_z()

        reward = machine.sample_reward(action)

        # update
        s[action] += reward
        n[action] += 1

        # recording
        actions[t] = action
        rewards[t] = reward
        probs[t] = machine.check_optimality(action)

    return rewards, probs
    pass
