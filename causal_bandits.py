import torch
import torch.distributions.beta as beta_dist


def sample_beta_dist(alpha, beta):
    return beta_dist.Beta(alpha, beta).sample()


def z_thompson(machine):
    s = torch.tensor([[1., 1.], [1., 1.]])
    n = torch.tensor([[1., 1.], [1., 1.]])
    time = machine.time_limit
    actions = [0] * time
    rewards = [0] * time
    probs = [0] * time

    for t in range(time):
        machine.set_time(t)

        # z-empowered thompson sampling action selection
        z = machine.get_hidden_z()
        val = torch.tensor([sample_beta_dist(s[z, 0], n[z, 0]),
                            sample_beta_dist(s[z, 1], n[z, 1])])
        action = val.argmax().item()

        reward = machine.sample_reward(action)

        # update
        s[z, action] += reward
        n[z, action] += 1 - reward

        # recording
        actions[t] = action
        rewards[t] = reward
        probs[t] = machine.check_optimality(action)

    return rewards, probs


def causal_thompson(machine):
    s = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    n = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    time = machine.time_limit
    actions = [0] * time
    rewards = [0] * time
    probs = [0] * time
    p_obs = machine.get_p_obs(num_obs=200)
    p_yx = [p_obs[0][1] / sum(p_obs[0]), p_obs[1][1] / sum(p_obs[1])]
    z_count = [0, 0]

    # seed P(y | do(X), z) with observations, whenever X = Z
    s[0, 0] = p_obs[0][1]
    s[1, 1] = p_obs[1][1]
    n[0, 0] = p_obs[0][0]
    n[1, 1] = p_obs[1][0]

    for t in range(time):
        machine.set_time(t)

        # causal thompson sampling action selection
        z = machine.get_hidden_z()
        z_prime = 1 - z
        z_count[z] += 1
        p_y_dox_z = [
            [s[0, 0] / (s[0, 0] + n[0, 0]), s[0, 1] / (s[0, 1] + n[0, 1])],
            [s[1, 0] / (s[1, 0] + n[1, 0]), s[1, 1] / (s[1, 1] + n[1, 1])]
        ]
        q1 = p_y_dox_z[z][z_prime].item()
        q2 = p_yx[z]
        w = [1, 1]
        w[z if q1 > q2 else z_prime] = 1 - abs(q1-q2)

        val = torch.tensor([sample_beta_dist(s[z, 0], n[z, 0]) * w[0],
                            sample_beta_dist(s[z, 1], n[z, 1]) * w[1]])
        action = val.argmax().item()

        reward = machine.sample_reward(action)

        # update
        s[z, action] += reward
        n[z, action] += 1 - reward

        # recording
        actions[t] = action
        rewards[t] = reward
        probs[t] = machine.check_optimality(action)

    return rewards, probs

