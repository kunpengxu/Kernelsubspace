
import numpy as np
import pandas as pd
import random
import Build_selfmatrix as build
from scipy import stats



def split(n, i):
    # method 1 :uniform
    rnd_array = np.random.multinomial(n, np.ones(i) / i, size=1)[0]

    # method 2 : ununiform
    while True:
        pick = random.sample(range(1, n), i)
        if np.sum(pick) == n:
            break
    rnd_array = pick

    return rnd_array


def get_timeseries_variation(split_series, indices, window_size):
    [Z, regime_num,D] = build.KSRM(split_series, indices, window_size)
    variation = max(regime_num) / window_size
    return [Z, variation,D]

def get_timeseries_variation2(split_series, indices, window_size):
    [Z, regime_num] = build.KSRM2(split_series, indices, window_size)

    return [Z]


def entropy_cut_off(scores):
    E = {}
    F = {}
    entropy_E = {}
    entropy_F = {}
    code = {}
    result = None
    for i in range(2, len(scores)):
        E_temp = scores[0:i]
        F_temp = scores[i:len(scores)]
        entropy_E[i - 1] = stats.entropy(E_temp) / len(E_temp)
        entropy_F[i] = stats.entropy(F_temp) / len(F_temp)
        E[i] = E_temp
        F[i] = F_temp
        code[i] = np.abs(entropy_E[i - 1] - entropy_F[i])
    minim = min(list(code.values()))
    for i in sorted(list(code.keys())):
        if code[i] == minim:
            result = (i, E[i], F[i], code[i])
            break
    return result[0] - 2


def generate_order(n_series, win_size, win_number, Noise_ratio):
    gnt = [0 for x in range(0, win_size)]
    Label = []
    regime_number = np.random.randint(3, 6)
    subspace = split(n_series, regime_number)
    r = random.sample(range(0, 5), regime_number)

    print("Regime_number=%d, subspace_size=" % regime_number, subspace)
    for j in range(1, regime_number + 1):

        Label.extend(list(np.full(subspace[j - 1], j - 1)))

        for n in range(subspace[j - 1]):

            globals()['gnt%s_%s' % (j, n)] = []

            for t in range(0, win_size):
                rg = fun(t)
                # f1 = math.cos(2 * math.pi * t / 5) + math.cos(math.pi * (t - 3))
                # f2 = math.sin(math.pi * t / 2 - 3) - math.sin(math.pi * t / 6)
                # f3 = math.tan(math.pi * t / 2 - 3) - 0.5 * math.cos(math.pi * (t - 3) / 6) + math.cos(
                #     math.pi * (t - 13))
                # f4 = math.sin(math.pi * t / 2 - 3) * math.cos(math.pi * (t - 3) / 6) * math.cos(math.pi * (t - 13))
                # f5 = math.cos(3 * math.pi * t / 5) + math.sin(2 * math.pi * t / 5 - t)
                index = r[j - 1]

                globals()['gnt%s_%s' % (j, n)].append(np.random.uniform(1 - Noise_ratio, 1) * rg[index])

            gnt = np.column_stack((gnt, globals()['gnt%s_%s' % (j, n)]))
    # gnt =np.row_stack((np.array(Label).T, gnt))

    gnt = pd.DataFrame(gnt);
    gnt = gnt.drop(gnt.iloc[:, 0], axis=1).T.reset_index(drop=True).T
    # gnt = gnt.drop(gnt.iloc[:, 1075:1301], axis=1).T.reset_index(drop=True).T
    return [gnt, pd.DataFrame(Label)]


def generate_order2(n_series, win_size, win_number, Noise_ratio):
    gnt = [0 for x in range(0, win_size)]
    Label = []
    regime_number = 5
    subspace = [100, 100, 100, 100, 100]
    r = random.sample(range(0, 5), regime_number)

    print("Regime_number=%d, subspace_size=" % regime_number, subspace)
    for j in range(1, regime_number + 1):

        Label.extend(list(np.full(subspace[j - 1], j - 1)))

        for n in range(subspace[j - 1]):

            globals()['gnt%s_%s' % (j, n)] = []

            for t in range(0, win_size):
                rg = fun(t)

                index = r[j - 1]

                globals()['gnt%s_%s' % (j, n)].append(np.random.uniform(1 - Noise_ratio, 1) * rg[index])

            gnt = np.column_stack((gnt, globals()['gnt%s_%s' % (j, n)]))


    gnt = pd.DataFrame(gnt);
    gnt = gnt.drop(gnt.iloc[:, 0], axis=1).T.reset_index(drop=True).T
    return [gnt, pd.DataFrame(Label)]


def generate_multi_win(n_series, win_size, win_number, Noise_ratio):
    gnt_multi = pd.DataFrame([0 for x in range(0, n_series)]).T
    for i in range(win_number):
        print('window' + str(i + 1) + '_size')

        [gnt_win, Label] = generate_order(n_series, win_size, win_number, Noise_ratio)

        gnt_multi = pd.concat([gnt_multi, gnt_win], axis=0, ignore_index=True)
    return [gnt_multi.drop(gnt_multi.iloc[0, :], axis=0).reset_index(drop=True), pd.DataFrame(Label)]



if __name__ == '__main__':
    [data, label] = generate_multi_win(500, 78, 3, 0.6)
