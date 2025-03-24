# SAE: Similarity Aware Evaluation (ICLR 2025 Oral)

Code for paper SAE: Rethinking the Generalization of Drug Target Affinity Prediction Algorithms via Similarity Aware Evaluation.
> [Rethinking the Generalization of Drug Target Affinity Prediction Algorithms via Similarity Aware Evaluation](https://openreview.net/forum?id=j7cyANIAxV)  \
> Authors: Chenbin Zhang, Zhiqiang Hu, Chuchu Jiang, Wen Chen, Jie Xu, Shaoting Zhang

Contact: chenbinzhang@moleculemind.com or huzq@pku.edu.cn. Feel free to ask any questions or discussions!

## Introduction
Similarity Aware Evaluation (SAE) is a novel train-test split methodology that can achieve desired similarity distribution via gradient descent.

## Requirements
```bash
python >= 3.7
torch >= 1.11
rdkit >= 2023.3.1
pandas
numpy
seaborn
matplotlib
omegaconf
```

To fully reproduce our splitting results, we can configure the environment as follows:
```bash
python==3.7.16
torch==1.11.0+cu113
numpy==1.21.6
rdkit==2023.3.1
```
And the remaining packages do not require specific versions.

## Quick start
You can split the dataset into training and testing sets by running the following command:

```bash
python SAE_split.py demo_configs/IC50_EGFR_balance.yaml
```

After running the command, the output files will be organized as follows:

```bash
demo_output/
    └── IC50_EGFR
        ├── info.csv
        └── sigmoid-True_base-lr-1.00e-02_optim-kind-ExtraAdam_sched-kind-CosineAnnealing_init-kind-custom_lamb-2.03e-03_sigma-0.10_scale-factor-100_bins-89e6b15907a9354ec4caf6e18ce378db_seed-233_init-scale-5_max-iters-20000
            ├── W.npy
            ├── real_R.npy
            ├── sim.png
            ├── test.csv
            ├── train.csv
            ├── viz_W.png
            └── viz_real_R.png
```

### Explanation of Output Files:
- **info.csv**: Contains metadata or additional information about the dataset split.
- **W.npy**: The resulting weight matrix.
- **real_R.npy**: The real-valued matrix generated during the splitting.
- **test.csv**: The resulting test set.
- **train.csv**: The resulting training set.
- **viz_W.png**: A visual representation of the weight matrix.
- **viz_real_R.png**: A visual representation of the similarity matrix.

## Grid-search for data splitting

The demo configuration file is `demo_configs/IC50_EGFR_mimic.yaml`, which is organized as follows: (The comments explain the optional values of the hyper-parameters.)
```yaml
dataset_path: $input_csv_path
save_dir: $save_dir
test_ratio: 0.2
hyper_param_dict:
    'bins': [ ]
    'seed': [233, ]
    'max_iters': [20000, ]
    'sigmoid': [True, ]  # {True, False}
    'base_lr': [1e-2, ]
    'optim_kind': ['ExtraAdam', ]  # {ExtraSGD, ExtraAdam, SGD, Adam, AdamW}
    'sched_kind': ['CosineAnnealing', ]  # {CosineAnnealing, CAWarmRStarts, Step}
    'init_kind': ['custom', ]  # {normal, uniform, custom}
    'lamb': [2.03091762e-03, ]
    'sigma': [0.1, ]
    'scale_factor': [100, ]
    'init_scale': [5, ]
```
The `bins` have two configuration options:

1. **Without Assigning Weights to Each Bin**: For example, to achieve a balanced split as described in the paper, such as "[0, 1/3, 2/3, 1]", we can set `bins` as `'bins': [[0, 1 / 3, 2 / 3, 1.0], ]` or `'bins': [[0, 0.33333, 0.66666, 1.0], ]`. For a "0.4-0.6" split, where the test distribution is expected to have its maximum similarity in the range between 0.4 and 0.6, we can configure `bins` as `'bins': [[0.4, 0.6], ]`.

2. **With Assigning Weights to Each Bin**: For a mimic split as discussed in the paper, we first calculate the bin count of the external test set. Then, we can configure `bins` as follows:
    ```yaml
    'bins': [
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0, 6, 36, 119, 247, 460, 271, 129, 52, 12]
        ],
    ]
    ```

## Other scripts
1. `select_split.py`
    - Description: select the optimal splitting result in the grid-search experiment.
    - Usage: `python select_split.py ${path to info.csv in the output directory of grid-search experiment}`
2. `double_check_split.py`
    - Description: double check the train-test similarity distribution.
    - Usage: `python double_check_split.py $dataset_dir $train.csv $test.csv $save_path`
    - Example: 
    ```bash
    python double_check_split.py demo_output/IC50_EGFR/sigmoid-True_base-lr-1.00e-02_optim-kind-ExtraAdam_sched-kind-CosineAnnealing_init-kind-custom_lamb-2.03e-03_sigma-0.10_scale-factor-100_bins-89e6b15907a9354ec4caf6e18ce378db_seed-233_init-scale-5_max-iters-20000/ train.csv test.csv ./viz.jpg
    ```


## Cite
If you use this code in your research, please cite the following paper:

```bibtex
@inproceedings{zhang2025rethinking,
    title={Rethinking the generalization of drug target affinity prediction algorithms via similarity aware evaluation},
    author={Chenbin Zhang and Zhiqiang Hu and Jiang Chuchu and Wen Chen and JIE XU and Shaoting Zhang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=j7cyANIAxV}
}
```
