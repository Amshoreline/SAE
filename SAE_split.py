# system
import os
import sys
import time
import itertools
import random
# torch,pandas,numpy
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
# plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# cooper: https://github.com/cooper-org/cooper/tree/master/cooper
import cooper
# utils
from utils import get_sim


class MaximumBalance(cooper.ConstrainedMinimizationProblem):

    def __init__(self, *, alpha, C, sigmas, S, E, lamb, scale_factor, eps=1e-6, sigmoid=False):
        '''
        Params:
            alpha (float):   proportion of the test set size
            C (tensor, K):   center point of each bin
            sigmas (tensor, K):  standard deviations of each bin
            E (tensor, K):   desired distribution over the bins, sum(E) == 1
            S (tensor, (N, N)):  similarity matrix for each pair of mols
            lamb (float):    weight of the regular loss
            scale_factor (float):   scaling factor of `log(sum(exp))` when calculating `R`
            eps (float):     epsilon
        '''
        super().__init__(is_constrained=True)
        self.alpha = alpha
        self.C = C
        self.sigmas = sigmas
        assert torch.abs(torch.sum(E) - 1) < eps
        self.E = E
        self.S = S
        self.lamb = lamb
        self.scale_factor = scale_factor
        self.eps = eps
        assert sigmoid in [True, False]
        self.sigmoid = sigmoid

    def closure(self, T):
        K = self.C.shape[0]  # number of bins
        N = T.shape[0]  # number of mols
        #
        if self.sigmoid:
            W = T.sigmoid()
            safe_W = W
            ineq_defect = None
        else:
            W = T
            safe_W = torch.clip(W, self.eps, 1 - self.eps)
            # Inequality constraints: w >= 0 (equiv. -w <= 0) and w <= 1 (equiv. w - 1 <= 0)
            ineq_defect = torch.stack([-W, W - 1])
        # Equality constraint
        eq_defect = torch.sum(W) - self.alpha * N
        #
        self.W = W
        # main_loss
        R = torch.log(
            torch.sum(torch.exp((1 - safe_W).view(1, -1) * self.S * self.scale_factor), dim=1)
        ) / self.scale_factor # shape=(N, ), Differentiable version of torch.max((1 - W).view(1, -1) * self.S, dim=1)
        # R = torch.max((1 - W).view(1, -1) * self.S, dim=1)[0]
        O = torch.sum(
            safe_W.view(-1, 1)
            * torch.softmax(
                -(R.view(-1, 1) - self.C.view(1, -1)) ** 2 / (2 * self.sigmas.view(1, -1) ** 2),
                dim=1
            ),
            dim=0
        )  # (K, )
        # main_loss = torch.sum((O - self.alpha * N / K) ** 2 / (self.alpha * N / K))
        obj_O = self.E * self.alpha * N
        with torch.no_grad():
            bin_mask = (obj_O > self.eps).detach()
        main_loss = torch.sum((O[bin_mask] - obj_O[bin_mask]) ** 2 / obj_O[bin_mask])
        #
        # regular_loss = torch.sum(
        #     -safe_W * torch.log(safe_W)
        #     -(1 - safe_W) * torch.log(1 - safe_W)
        # )
        regular_loss = F.binary_cross_entropy(safe_W, safe_W, reduction='sum')
        #
        loss = main_loss + self.lamb * regular_loss
        # For debug
        self.R = R
        self.O = O
        self.main_loss = main_loss.item()
        self.regular_loss = regular_loss.item()
        self.total_loss = loss.item()
        #
        return cooper.CMPState(loss=loss, eq_defect=eq_defect, ineq_defect=ineq_defect)


class Scheduler:

    def __init__(self, sched_class, sched_kwargs):
        self.sched_class = sched_class
        self.sched_kwargs = sched_kwargs
    
    def __call__(self, optimizer):
        return self.sched_class(optimizer, **self.sched_kwargs)


def init_custom(T, train_ratio, inv_sigmoid=False, init_scale=5):
    assert inv_sigmoid in [True, False]
    if inv_sigmoid:
        pos_val = init_scale
        neg_val = -init_scale
    else:
        pos_val = 1
        neg_val = 0
    #
    inds = np.arange(T.shape[0])
    random.shuffle(inds)
    with torch.no_grad():
        T[inds[: int(len(inds) * train_ratio)]] = neg_val  # Train
        T[inds[int(len(inds) * train_ratio) :]] = pos_val  # Val+Test
    return T


def init_normal(T, inv_sigmoid=False, init_scale=5):
    T = torch.nn.init.normal_(T)  # ()
    assert inv_sigmoid in [True, False]
    with torch.no_grad():
        if inv_sigmoid:
            T.mul_(init_scale).clamp_(min=-5, max=5)
        else:
            T.add_(1).clamp_(min=0, max=1)
    return T


def init_uniform(T, inv_sigmoid=False, init_scale=5):
    T = torch.nn.init.uniform_(T)
    assert inv_sigmoid in [True, False]
    with torch.no_grad():
        if inv_sigmoid:
            T.add_(-0.5).mul_(2 * init_scale)
    return T


def balance_split(
        table, S, alpha, max_iters=5000, bins=[0, 1/3, 2/3, 1], sigma=0.01, lamb=1.0, scale_factor=100,
        base_lr=1e-3, optim_kind='ExtraSGD', sched_kind='CosineAnnealing', init_kind='normal', eps=1e-6,
        sigmoid=True, init_scale=5,
        seed=233,
    ):
    assert 0. < alpha < 1.
    # seed all
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # experimental configurations
    # log_freq = max(max_iters // 100, 1)
    log_freq = 500
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('\tUse device', device)
    # get optimizer, schedulers and init_func
    optim_class, optim_kwargs = {
        'ExtraSGD': (cooper.optim.ExtraSGD, {'momentum': 0.7}),
        'ExtraAdam': (cooper.optim.ExtraAdam, {'weight_decay': 1e-5}),
        'SGD': (torch.optim.SGD, {'momentum': 0.7}),
        'Adam': (torch.optim.Adam, {'weight_decay': 1e-5}),
        'AdamW': (torch.optim.AdamW, {}),
    }[optim_kind]
    sched_class, sched_kwargs = {
        'CosineAnnealing': (torch.optim.lr_scheduler.CosineAnnealingLR, {'T_max': max_iters}),
        'CAWarmRStarts': (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, {'T_0': max_iters // 2}),
        'Step': (torch.optim.lr_scheduler.StepLR, {'step_size': max_iters // 4}),
    }[sched_kind]
    dual_scheduler = Scheduler(sched_class, sched_kwargs)
    init_func, init_kwargs = {
        'normal': (init_normal, {'inv_sigmoid': sigmoid, 'init_scale': init_scale}),
        'uniform': (init_uniform, {'inv_sigmoid': sigmoid, 'init_scale': init_scale}),
        'custom': (init_custom, {'train_ratio': 1 - alpha, 'inv_sigmoid': sigmoid, 'init_scale': init_scale})
    }[init_kind]
    # basic coefficients
    inds = table.index
    N = len(table)
    num_train_mols = int((1 - alpha) * N)
    num_test_mols = N - num_train_mols
    if isinstance(bins, tuple):
        bins, E = bins
    else:
        E = [1] * (len(bins) - 1)
    if bins[0] > (0 + eps):
        bins = [0] + bins
        E.insert(0, 0)
    if bins[-1] < (1 - eps):
        bins = bins + [1]
        E.append(0)
    E = torch.tensor(E).to(device)
    E = E / torch.sum(E)
    K = len(bins) - 1
    bins_arr = bins
    bins = torch.tensor(bins).to(device)
    bins[-1] += eps
    C = (bins[1 :] + bins[: -1]) / 2
    # sigmas = torch.tensor([sigma] * K).to(device)  # constant value
    sigmas = (bins[1 :] - bins[: -1]) * sigma
    S_arr = S
    S = torch.tensor(S).to(device)
    # build cooper
    cmp_ = MaximumBalance(
        alpha=alpha, C=C, sigmas=sigmas, S=S, E=E, lamb=lamb, scale_factor=scale_factor,
        eps=eps, sigmoid=sigmoid
    )
    formulation = cooper.LagrangianFormulation(cmp_)
    T = torch.nn.Parameter(torch.zeros(N, requires_grad=True, device=device))
    init_func(T, **init_kwargs)
    primal_optimizer = optim_class([T], lr=base_lr, **optim_kwargs)
    dual_optimizer = cooper.optim.partial_optimizer(optim_class, lr=base_lr, **optim_kwargs)
    coop = cooper.ConstrainedOptimizer(formulation, primal_optimizer, dual_optimizer, dual_scheduler)
    # training
    best_res = None
    best_score = -np.inf
    start_time = time.time()
    for iter_ in range(max_iters):
        coop.zero_grad()
        lagrangian = formulation.composite_objective(cmp_.closure, T)
        formulation.custom_backward(lagrangian)
        coop.step(cmp_.closure, T)
        # calculation of intermediate results
        if ((iter_) % log_freq == 0) or (iter_ == max_iters - 1):
            W = cmp_.W.detach().cpu().numpy()
            cur_sum = np.sum(W)
            bins_info = ''
            R = cmp_.R.detach().cpu().numpy()
            R_hist = []
            hard_R = np.max((1 - W).reshape(1, -1) * S_arr, axis=1)
            hard_R_hist = []
            part_res = np.argpartition(W, num_train_mols)
            real_R = np.max(S_arr[inds[part_res[num_train_mols :]]][:, inds[part_res[: num_train_mols]]], axis=1)
            real_R_hist = []
            for bin_left, bin_right in zip(bins_arr, bins_arr[1 :] + [np.inf]):
                bins_info += f' [{bin_left:.2f}, {bin_right:.2f}]'
                R_hist.append(round(np.sum(W * (R >= bin_left) * (R < bin_right)), 1))
                hard_R_hist.append(round(np.sum(W * (hard_R >= bin_left) * (hard_R < bin_right)), 1))
                real_R_hist.append(round(np.sum((real_R >= bin_left) * (real_R < bin_right)), 1))
            real_R_hist[-2] += real_R_hist[-1]
            real_R_hist = real_R_hist[: -1]
            #
            score = -np.mean(np.abs(np.array(real_R_hist) / num_test_mols - E.cpu().numpy()))
            res = (
                part_res,
                [cmp_.main_loss, cmp_.regular_loss, cmp_.total_loss],
                cur_sum, W, real_R, real_R_hist, score
            )
            if best_score < score:
                best_score = score
                best_res = res
            #
            print(
                f'\tIter [{iter_}/{max_iters}] '
                f'Lag {round(lagrangian.item(), 3)} '
                f'MainLoss: {cmp_.main_loss:.2e} RegularLoss: {cmp_.regular_loss:.2e} '
                f'cur_sum {cur_sum:.2f}/{alpha * N:.2f} '
                f'LR {coop.primal_optimizer.param_groups[0]["lr"]:.2e} {coop.dual_optimizer.param_groups[0]["lr"]:.3e} '
                f'Time {time.time() - start_time:.1f} \n'
                f'\t\tO: {np.round(cmp_.O.detach().cpu().numpy(), 2).tolist()}\n'
                f'\t\tbins:\t{bins_info}\n'
                f'\t\tR:\t{R_hist}\n'
                f'\t\thard_R:\t{hard_R_hist}\n'
                f'\t\treal_R:\t{real_R_hist} score:{score:.3f}'
            )
    part_res, *other_res_list = best_res
    return (
        table.loc[inds[part_res[: num_train_mols]]],  # train+val table
        table.loc[inds[part_res[num_train_mols :]]],  # test table
        *other_res_list
    )



def main(dataset_path=sys.argv[1], save_dir=sys.argv[2]):
    test_ratio = 0.2
    #
    print(dataset_path)
    end_time = time.time()
    #
    data_table = pd.read_csv(dataset_path, header=None)
    smis = data_table[0].tolist()
    S = get_sim(smis, smis) * (1 - np.eye(len(data_table)))
    #
    dataset_name = os.path.basename(dataset_path).replace('.csv', '')
    print('Time consuming for similarity matrix is:', int(time.time() - end_time))
    # balance split
    hyper_param_dict = {
        'sigmoid': [True, ],
        'base_lr': [1e-2, ],
        'optim_kind': ['ExtraAdam', ],
        'sched_kind': ['CosineAnnealing', ],
        'init_kind': ['custom', ],
        'lamb': [2.03e-03, ],
        'sigma': [0.1, ],
        'scale_factor': [100, ],
        'bins': [[0, 1 / 3, 2 / 3, 1.0], ],
        'seed': [233, ],
        'init_scale': [5, ],
        'max_iters': [20000, ],
        # 'repeat_ind': range(16), 
    }
    # prepare info_table
    info_table = {
        'dirname': [],
        'main loss': [], 'regular loss': [], 'total loss': [],
        'test ratio': [], 'sum(W)': [], 'N': [], 'test hist': [], 'hist score': [],
    }
    info_table.update({key : [] for key in hyper_param_dict.keys()})
    #
    groups = list(itertools.product(*hyper_param_dict.values()))
    print('#groups =', len(groups))
    for group in groups:
        group_dict = dict(zip(hyper_param_dict.keys(), group))
        print()
        print(group_dict)
        param_dict = group_dict.copy()
        if 'repeat_ind' in param_dict:
            del param_dict['repeat_ind']
        # *train/val+test split*
        train_table, test_table, loss_list, test_count, W, real_R, real_R_hist, hist_score = balance_split(
            data_table, S, test_ratio,
            **param_dict,
        )
        print(f'\tbest real_R:\t{real_R_hist}')
        # train/val split, stratified sampling
        num_vals = int(len(data_table) * (1 - test_ratio)) - int(len(data_table) * (1 - test_ratio))
        # save dataset
        str_group_dict = []
        for key, value in group_dict.items():
            str_key = key.replace('_', '-')
            if key in ['lamb', 'base_lr']:
                str_value = f'{value:.2e}'
            elif isinstance(value, float):
                str_value = f'{value:.2f}'
            elif isinstance(value, list):
                str_value = '[' + ','.join([f'{item:.2f}' for item in value]) + ']'
            elif isinstance(value, tuple):
                str_value = '[' + ','.join([f'{item:.2f}' for item in value[0]]) + ']'
            else:
                str_value = str(value)
            str_group_dict.append(f'{str_key}-{str_value}')
        dirname = '_'.join(str_group_dict)
        data_dir = f'{save_dir}/{dataset_name}/{dirname}'
        print('\tSave results to', dirname)
        os.makedirs(data_dir, exist_ok=True)
        train_table.to_csv(f'{data_dir}/train.csv', index=False, header=False)
        test_table.to_csv(f'{data_dir}/test.csv', index=False, header=False)
        # write information to info_table
        for key in group_dict.keys():
            info_table[key].append(group_dict[key])
        info_table['dirname'].append(dirname)
        info_table['main loss'].append(loss_list[0])
        info_table['regular loss'].append(loss_list[1])
        info_table['total loss'].append(loss_list[2])
        info_table['test ratio'].append(test_count / len(W))
        info_table['sum(W)'].append(test_count)
        info_table['N'].append(len(W))
        info_table['test hist'].append(real_R_hist)
        info_table['hist score'].append(hist_score)
        # save W
        plt.title(f'lamb: {group_dict["lamb"]:.2e}')
        plt.ylim(0, len(data_table))
        plt.hist(W, bins=20)
        plt.savefig(f'{data_dir}/viz_W.png')
        plt.close('all')
        np.save(f'{data_dir}/W.npy', W)
        # save real_R
        plt.hist(x=real_R, bins=np.linspace(0, 1, 21))
        plt.savefig(f'{data_dir}/viz_real_R.png')
        plt.close('all')
        np.save(f'{data_dir}/real_R.npy', real_R)
        pd.DataFrame(info_table).to_csv(f'{save_dir}/{dataset_name}/info.csv')                        


if __name__ == '__main__':
    main()
