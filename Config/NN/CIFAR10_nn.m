dataset = 'CIFAR10';  % Xtrain, ytrain, Xtest, ytest, classes, optimal_value
model_radius_W = 10.0;
model_radius_b = 10.0;
model_dim_hidden = 10;

IS_SFO_FIXED = true;

IS_ADVERSARIAL = false;

IS_CALCULATING_REGRET = false;

selected_methods = {'ORGFW', 'OSFW', 'OFW', 'OAW', 'ROFW', 'FW', 'SVRG_FW', 'SPIDER_FW'};

batch_size = 16;

print_freq = 1e2;  % the frequency of progress evaluation, e.g., evaluate the loss function after every 100 iterations
num_iters_base = 1.5e4 * 6e0 * 0.5;  % number of iterations of the baseline method (OSFW)

print_freq_offline = 1e3;  % SPIDER or SVRG
num_iters_base_offline = 7.5e4 * 6e0;  % SPIDER or SVRG

print_freq_batch = 1e1;  % the frequency of progress evaluation, e.g., evaluate the loss function after every 100 iterations
num_iters_base_batch = 1e2 * 4;  % number of iterations of the baseline method (FW)

eta_coef_ORGFW = 1e-1;
eta_exp_ORGFW = 2/3;
rho_coef_ORGFW = 2e-0;
rho_exp_ORGFW = 2/3;

eta_coef_OSFW = 1e-1;
eta_exp_OSFW = 2/3;
rho_coef_OSFW = 1e-0;
rho_exp_OSFW = 2/3;

eta_coef_FW = 2.5e-1;
eta_exp_FW = 1;

eta_coef_OFW = 1e-0;
eta_exp_OFW = 1;

eta_coef_ROFW = 1e-1;
eta_exp_ROFW = 1/2;
reg_coef_ROFW = 1e3;

eta_coef_OAW = 1e-0;
eta_exp_OAW = 1;

eta_coef_SPIDER = 5e-4;
eta_exp_SPIDER = 0.0;
epoch_length_SPIDER = 200;
batch_size_SPIDER = 16;


eta_coef_SVRG = 1e-3;
eta_exp_SVRG = 0.0;
epoch_length_SVRG = 200;
batch_size_SVRG = 50;
