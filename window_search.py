import numpy as np

from tqdm import tqdm
import generate_data_Multi_windows as gen
import os
from copy import deepcopy
import json
import plotting as xplt

wrt = 'highest'


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class Winsearch:

    def __init__(self, data=None, len_series=None, win_start=5, win_end=None, root=None, file_name=None):
        self.data = data
        self.len_series = len_series
        self.win_start = win_start
        self.win_end = len(self.data)//3
        self.root = root
        self.file_name = file_name

    # @staticmethod
    # def get_timeseries_variation(self, split_series, indices, window_size):
    #     [Z,regime_num] = build.KSRM(split_series, indices, window_size)
    #     variation = max(regime_num) / window_size
    #
    #     return [Z,variation]

    def win_number(self, scale):
        # data = self.data
        subdir = self.file_name.split('.')[0].upper()
        subdir1 = self.file_name.split('.')[0]
        if 'results' not in os.listdir(self.root):
            os.mkdir(self.root + 'results/')
        if subdir not in os.listdir(self.root + 'results/'):
            os.makedirs(self.root + 'results/' + subdir + '/')
        if 'images' not in os.listdir(self.root):
            os.mkdir(self.root + 'images/')

        path_to_json = self.root + 'results/' + subdir + '/'
        path_to_store = self.root + 'images/'
        name = subdir1

        data_variation = {}
        data_KSRM = {}
        if 'data_variation_scale_' + scale + '.json' not in os.listdir(path_to_json):
            # print('Predefining window sizes...')
            # chunk_sizes = [i for i in range(2, len(timeseries) // 3)]

            print('scanning data for \n window_size = ')
            current_variation = None
            for window_size in tqdm(range(self.win_start, self.win_end)):
                print(str(window_size), end=',')

                indices = range(1 * window_size, (len(self.data) // window_size + 1) * window_size, window_size)
                split_series = np.split(self.data, indices)
                [Z, variation] = gen.get_timeseries_variation(split_series, indices, window_size)
                data_KSRM[str(window_size)] = Z
                data_variation[str(window_size)] = variation
                if current_variation is None:
                    current_variation = deepcopy(variation)
                else:
                    diff = np.abs(variation - current_variation)
                    if variation <= .001:
                        break
                    # if diff <= .0001: break
                    else:
                        current_variation = deepcopy(variation)
                with open(path_to_json + 'data_variation_scale_' + scale + '.json', 'w') as outputfile:
                    json.dump(data_variation, outputfile, indent=3)

        else:
            with open(path_to_json + 'data_variation_scale_' + scale + '.json', 'r') as inputfile:
                data_variation = json.load(inputfile)

        if len(data_variation) < self.win_end - self.win_start:
            data_key = list(data_variation.keys())
            print("start from window_size=" + str(int(data_key[len(data_key)-1])+1))
            current_variation = None
            for window_size in tqdm(range(int(data_key[len(data_key)-1])+1, self.win_end)):
                print(str(window_size), end=',')

                indices = range(1 * window_size, (len(self.data) // window_size + 1) * window_size, window_size)
                split_series = np.split(self.data, indices)
                window_stamps = []
                for i in range(len(indices)):
                    window_stamps.append('$W^' + str(window_size) + '_' + str(i + 1) + '$')

                [Z, variation] = gen.get_timeseries_variation(split_series, indices, window_size)
                data_KSRM[str(window_size)] = Z
                data_variation[str(window_size)] = variation
                if current_variation is None:
                    current_variation = deepcopy(variation)
                else:
                    diff = np.abs(variation - current_variation)
                    if variation <= .001:
                        break
                    # if diff <= .0001: break
                    else:
                        current_variation = deepcopy(variation)
                with open(path_to_json + 'data_variation_scale_' + scale + '.json', 'w') as outputfile:
                    json.dump(data_variation, outputfile, indent=3)
        else:
            data_variation = data_variation

        temp_dico = {}
        for k, v in data_variation.items(): temp_dico[int(k)] = v
        keys = sorted(list(temp_dico.keys()))
        dynamic = {}
        variation = temp_dico[keys[0]]
        dynamic[keys[0]] = temp_dico[keys[0]]
        for j in range(1, len(keys)):
            current_variation = temp_dico[keys[j]]
            dynamic[keys[j]] = current_variation
            if np.abs(current_variation - variation) < .0001:
                break
            else:
                variation = deepcopy(current_variation)

        # dynamic = {}
        # for k,v in data_variation.items(): dynamic[int(k)] = v
        window_sizes = sorted(list(dynamic.keys()))

        mld_cut = gen.entropy_cut_off(np.array([dynamic[s] for s in window_sizes]))
        tresh = mld_cut
        print('window_size='+str(tresh))
        xplt.series_variation(dynamic=dynamic, window_sizes=window_sizes, name=name, root=self.root)
        KSRM = {}
        return [tresh, KSRM]
