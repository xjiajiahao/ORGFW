### Generate datasets
1. Download the MNIST and CIFAR-10 datasets. See `Data/MNIST/README.md` and `Data/CIFAR-10/README.md` for details.
2. Launch `MATLAB` from the `Utils` directory, and the run `LoadMNIST.m` and `LoadCifar10.m` to generate the dataset files.

### How to Run
1. To conduct the multiclass logistic regression experiment, first open `main_LR.m` and choose one configuration script file, for example,
``` matlab
%% Section 2: Set parameters (choose one configuration script and comment the others, the configuration scripts can be found in the Config/LR/ folder)
if ~exist('IS_TUNING_PARAMETERS', 'var') || IS_TUNING_PARAMETERS == false
    MNIST_stochastic_regret;
    % MNIST_stochastic_time;
    % MNIST_adversary;
    % CIFAR10_stochastic_regret;
    % CIFAR10_stochastic_time;
    % CIFAR10_adversary;
end
```
After saving changes to `main_LR.m`, type `main_LR` in the MATLAB command window.

2. To conduct the one-hidden-layer neural network experiment, first open `main_NN.m` and choose one configuration script file, for example,
``` matlab
%% Section 2: Set parameters (choose one configuration script and comment the others, the configuration scripts can be found in the Config/LR/ folder)
if ~exist('IS_TUNING_PARAMETERS', 'var') || IS_TUNING_PARAMETERS == false
    MNIST_NN;
    % CIFAR10_NN;
end
```
After saving changes to `main_NN.m`, type `main_NN` in the MATLAB command window.

### Project Structure
```
├── Algorithms
│   ├── README.m
│   ├── cell_array
│   │   ├── FW.m
│   │   ├── OAW.m
│   │   ├── OFW.m
│   │   ├── ORGFW.m
│   │   ├── OSFW.m
│   │   ├── ROFW.m
│   │   ├── SPIDER_FW.m
│   │   └── SVRG_FW.m
│   └── vector
│       ├── FW.m
│       ├── MFW.m
│       ├── MORGFW.m
│       ├── OAW.m
│       ├── OFW.m
│       ├── ORGFW.m
│       ├── OSFW.m
│       └── ROFW.m
├── Config  # configuration scripts
│   ├── LR
│   │   ├── CIFAR10_adversary.m  # CIFAR-10 dataset, adversarial online setting
│   │   ├── CIFAR10_stochastic_regret.m  # CIFAR-10 dataset, stochastic online setting, report regret v.s. #rounds
│   │   ├── CIFAR10_stochastic_time.m  # CIFAR-10 dataset, stochastic optimization setting, report suboptimality v.s. running time
│   │   ├── MNIST_adversary.m  # MNIST dataset, adversarial online setting
│   │   ├── MNIST_stochastic_regret.m  # MNIST dataset, stochastic online setting, report regret v.s. #rounds
│   │   └── MNIST_stochastic_time.m  # MNIST dataset, stochastic optimization setting, report suboptimality v.s. running time
│   └── NN
│       ├── CIFAR10_NN.m  # CIFAR-10 dataset, stochastic optimization setting, report suboptimality v.s. running time
│       └── MNIST_NN.m  # MNIST dataset, stochastic optimization setting, report suboptimality v.s. running time
├── Data
│   ├── CIFAR-10
│   │   └── README.md
│   ├── CIFAR10_LR_opt.mat
│   ├── CIFAR10_NN_opt.mat
│   ├── MNIST
│   │   └── README.md
│   ├── MNIST_LR_opt.mat
│   └── MNIST_NN_opt.mat
├── README.md
├── Utils
│   ├── LoadCifar10.m
│   └── LoadMNIST.m
├── main_LR.m
└── main_NN.m
```
