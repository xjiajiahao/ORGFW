dataset = 'CIFAR10';  % Xtrain, ytrain, Xtest, ytest, classes, optimal_value
model_radius = 32.0;

IS_SFO_FIXED = false;

IS_ADVERSARIAL = false;

IS_CALCULATING_REGRET = true;

selected_methods = {'ORGFW', 'OSFW', 'OFW', 'ROFW', 'OAW'};

batch_size = 5e2;
sub_batch_size = 5e2;

print_freq = 1e0;  % regret
num_iters_base = 1e2;  % regret

eta_coef_ORGFW = 1e-1;
eta_exp_ORGFW = 2/3;
rho_coef_ORGFW = 1e0;
rho_exp_ORGFW = 2/3;

eta_coef_OSFW = 1e-1;
eta_exp_OSFW = 2/3;
rho_coef_OSFW = 1e0;
rho_exp_OSFW = 1/2;

eta_coef_FW = 2.5e-1;
eta_exp_FW = 1.0;

eta_coef_OFW = 5e-2;
eta_exp_OFW = 1/2;

eta_coef_ROFW = 0.6e-1;
eta_exp_ROFW = 2/3;
reg_coef_ROFW = 1e3;

eta_coef_OAW = 5e-2;
eta_exp_OAW = 5e-1;
