import multiprocessing
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import torch
import tqdm
import bandits
import causal_bandits
import BanditMachine


def get_bandit_algo(algo_name):
    if algo_name == 'ucb':
        return bandits.ucb
    if algo_name == 'thompson':
        return bandits.thompson
    if algo_name == 'e-greedy':
        return bandits.epsilon_greedy
    if algo_name == 'observation':
        return bandits.observational
    if algo_name == 'gittins':
        return bandits.gittins
    if algo_name == 'exp3':
        return bandits.exp3
    if algo_name == 'z_thompson':
        return causal_bandits.z_thompson
    if algo_name == 'causal_thompson':
        return causal_bandits.causal_thompson


def compare_bandits(algos, machine_type):
    machine = BanditMachine.BanditMachine(machine_type, time_limit=1000)
    opt_total = []
    reward_total = []
    machine.reset_randomness()
    for m in range(len(algos)):
        machine.reset()
        with torch.no_grad():
            bandit_func = get_bandit_algo(algo_name=algos[m])
            rewards, opt = bandit_func(machine)
            opt_total.append(opt)
            reward_total.append(rewards)

    return opt_total, reward_total


def compare_bandits_multiple_runs(algos, machine_type, num_runs=10):
    num_cores = multiprocessing.cpu_count()
    r_list = Parallel(n_jobs=num_cores)(
        delayed(compare_bandits)(algos, machine_type)
        for _ in tqdm.tqdm(range(num_runs)))

    opt_total = torch.tensor(
        [r_list[i][0] for i in range(len(r_list))]).float().sum(dim=0)

    opt_total /= num_runs
    x = opt_total.numpy().tolist()

    # plotting
    color_list = ['black', 'gray', 'brown', 'tomato', 'plum']
    for i in range(len(algos)):
        plt.plot([j for j in range(len(x[i]))], x[i],
                 color=color_list[i], label=algos[i],
                 linewidth=0.5)
    plt.legend()
    plt.title(machine_type)
    # plt.show()
    plt.savefig('{}_{}_{}.png'.format(machine_type, num_runs, len(algos)))
    plt.close()
    return


def main():
    algos = ['thompson', 'z_thompson', 'causal_thompson']
    print('comparing:', algos)
    for m_type in ['normal', 'greedy', 'generous', 'paradoxical', 'switch',
                   'inevitable']:
        print('- running {} bandit setting on {} cores:'.format(
            m_type, multiprocessing.cpu_count()))
        compare_bandits_multiple_runs(algos, m_type, 1000)

    #
    algos = ['thompson', 'ucb', 'gittins', 'exp3', 'e-greedy']
    print('comparing:', algos)
    print('- running {} bandit setting on {} cores:'.format(
        'normal', multiprocessing.cpu_count()))
    compare_bandits_multiple_runs(algos, 'normal', 1000)


if __name__ == '__main__':
    main()

