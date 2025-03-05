# Similarity Aware Evaluation (SAE)

[ICLR 2025 Oral](https://openreview.net/forum?id=j7cyANIAxV)

## Introduction
Similarity Aware Evaluation (SAE) is a framework that optimizes train-test splits via gradient descent to enforce customizable similarity distributions.

## Requirements
```bash
python >= 3.7
torch >= 1.11
rdkit >= 2023.3.1
pandas
numpy
seaborn
matplotlib
```

## Quick start
You can split the dataset into training and testing sets by running the following command:

```bash
python SAE_split.py demo_input/IC50_EGFR.csv demo_output
```

After running the command, the output files will be organized as follows:

```bash
demo_output/
    └── IC50_EGFR
        ├── info.csv
        └── sigmoid-True_base-lr-1.00e-02_optim-kind-ExtraAdam_sched-kind-CosineAnnealing_init-kind-custom_lamb-2.03e-03_sigma-0.10_scale-factor-100_bins-[0.00,0.33,0.67,1.00]_seed-233_init-scale-5_max-iters-20000
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
Coming soon. (If you are in a hurry, please email chenbinzhang@moleculemind.com)

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
