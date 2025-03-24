import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#
from utils import get_sim

sns.set_style('whitegrid')


def main(data_dir, train_files, test_files, save_path):
    if os.path.isdir(f'{data_dir}/{train_files}'):
        os.makedirs(save_path, exist_ok=True)
        filenames = sorted(os.listdir(f'{data_dir}/{train_files}'))
        for filename in filenames:
            if os.path.exists(f'{data_dir}/{train_files}/{filename}') and os.path.exists(f'{data_dir}/{test_files}/{filename}'):
                main(data_dir, f'{train_files}/{filename}', f'{test_files}/{filename}', f'{save_path}/{filename}'.replace('.csv', '.png'))
        return
    print('\tTrain:', train_files, '\n\tTest: ', test_files)
    train_files = train_files.split('+')
    test_files = test_files.split('+')
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    assert save_path.endswith('.png') or save_path.endswith('.jpg')
    # Get similarities
    train_smis = np.concatenate([pd.read_csv(f'{data_dir}/{data_file}', header=None)[0].to_numpy() for data_file in train_files])
    test_smis = np.concatenate([pd.read_csv(f'{data_dir}/{data_file}', header=None)[0].to_numpy() for data_file in test_files])
    S = get_sim(test_smis, train_smis)
    assert S.shape[0] == len(test_smis)
    assert S.shape[1] == len(train_smis)
    max_sim_list = np.max(S, axis=1)  # shape = (#test_smis, )
    # Plot
    bins = [0, 1 / 3, 2 / 3, 1.01]
    bin_counts = []
    for bin_left, bin_right in zip(bins[: -1], bins[1 :]):
        bin_counts.append(np.sum((max_sim_list >= bin_left) * (max_sim_list < bin_right)))
    bin_counts[-1] += np.sum(max_sim_list == bins[-1])
    print(bin_counts, np.sum(bin_counts))
    assert np.sum(bin_counts) == S.shape[0]
    plt.title(f'Train: {S.shape[0]} Test: {S.shape[1]} Bins: {bin_counts}')
    plt.hist(
        max_sim_list,
        bins=bins,
        # cumulative=True, density=True, histtype='step'
    )
    plt.savefig(save_path)
    plt.close('all')


if __name__ == '__main__':
    main(
        data_dir=sys.argv[1], train_files=sys.argv[2], test_files=sys.argv[3],
        save_path=sys.argv[4]
    )
