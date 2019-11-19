dataset = 'MNIST';  % Xtrain, ytrain, Xtest, ytest, classes, optimal_value
model_radius = 8.0;

IS_SFO_FIXED = false;

IS_ADVERSARIAL = true;

IS_CALCULATING_REGRET = true;

selected_methods = {'MORGFW', 'OFW', 'OAW', 'ROFW', 'MFW'};

batch_size = 6e2;
sub_batch_size = 4;
print_freq = 1e0;  % the frequency of progress evaluation, e.g., evaluate the loss function after every 100 iterations
num_iters_base = 1e2;  % number of iterations of the baseline method (OSFW)
% num_iters_base = ceil(32e2 * 6e4 / batch_size / 2);  % number of iterations of the baseline method (OSFW)

eta_coef_ORGFW = 1.25e-0;
eta_exp_ORGFW = 1;
rho_coef_ORGFW = 0.75e-0;
rho_exp_ORGFW = 2/3;

eta_coef_OSFW = 1e-0;
eta_exp_OSFW = 1;
rho_coef_OSFW = 1e-0;
rho_exp_OSFW = 2/3;

eta_coef_FW = 1.5e-0;
eta_exp_FW = 1.0;

eta_coef_OFW = 1e0;
eta_exp_OFW = 1/2;

eta_coef_ROFW = 2;
eta_exp_ROFW = 0.5;
reg_coef_ROFW = 1e4;

eta_coef_OAW = 1e0;
eta_exp_OAW = 1/2;

eta_coef_MFW = 1e-0;
eta_exp_MFW = 1;
rho_coef_MFW = 4e-0;
rho_exp_MFW = 1/2;
reg_coef_MFW = 2e1;

eta_coef_MORGFW = 1e-0;
eta_exp_MORGFW = 1;
rho_coef_MORGFW = 2e-0;
rho_exp_MORGFW = 1/2;
reg_coef_MORGFW = 10e1;
