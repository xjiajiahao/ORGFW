%% Section 1: Include necessary files
addpath('./Algorithms/cell_array/');
addpath('./Config/NN/');
DATA_ROOT = './Data/';

%% Section 2: Set parameters (choose one configuration script and comment the others, the configuration scripts can be found in the Config/LR/ folder)
if ~exist('IS_TUNING_PARAMETERS', 'var') || IS_TUNING_PARAMETERS == false
    MNIST_NN;
    % CIFAR10_NN;
end

%% Section 3: Load data
data_file_name = [DATA_ROOT, dataset, '_dataset.mat'];
load(data_file_name, 'Xtrain', 'ytrain', 'Xtest', 'ytest');
[dimension, data_size] = size(Xtrain);
num_classes = length(unique(ytrain));

optimal_solution = [];
optimal_value_file_name = [DATA_ROOT, dataset, '_NN_opt.mat'];
% if isfile(optimal_value_file_name)
if exist(optimal_value_file_name, 'file') == 2
    load(optimal_value_file_name, 'optimal_value', 'optimal_solution');
    IS_OPTIMAL_VALUE_AVAILABLE = true;
else
    IS_OPTIMAL_VALUE_AVAILABLE = false;
end

if ~IS_OPTIMAL_VALUE_AVAILABLE && IS_CALCULATING_REGRET
    IS_CALCULATING_REGRET = false;
end


%% Section 4: Run algorithms, do NOT modify the code below
if IS_SFO_FIXED
    num_iters_ORGFW = num_iters_base;
    num_iters_OSFW = num_iters_base * 2;
    num_iters_OFW = ceil(sqrt(num_iters_base) * 2);
    print_freq_OFW = ceil(num_iters_OFW / num_iters_base * print_freq);
    num_iters_OAW = ceil(sqrt(num_iters_base) * 2);
    print_freq_OAW = ceil(num_iters_OAW / num_iters_base * print_freq);
    num_iters_ROFW = num_iters_base * 2;
    print_freq_FW = print_freq_batch;
    if any(strcmp(selected_methods, 'SPIDER_FW'))
        num_iters_SPIDER = ceil(num_iters_base_offline * 2 * batch_size / ((epoch_length_SPIDER - 1) * batch_size_SPIDER * 2 + data_size) * epoch_length_SPIDER);
        print_freq_SPIDER = ceil(num_iters_SPIDER / num_iters_base_offline * print_freq_offline);
    end
    if any(strcmp(selected_methods, 'SVRG_FW'))
        num_iters_SVRG = ceil(num_iters_base_offline * 2  * batch_size/ ((epoch_length_SVRG - 1) * batch_size_SVRG * 2 + data_size) * epoch_length_SVRG);
        print_freq_SVRG = ceil(num_iters_SVRG / num_iters_base_offline * print_freq_offline);
    end
else
    num_iters_ORGFW = num_iters_base;
    num_iters_OSFW = num_iters_base;
    num_iters_OFW = num_iters_base;
    print_freq_OFW = print_freq;
    num_iters_ROFW = num_iters_base;
    num_iters_OAW = num_iters_base;
    print_freq_OAW = print_freq;
    num_iters_MFW = num_iters_base;
    num_iters_MORGFW = num_iters_base;
    print_freq_FW = print_freq_batch;
    num_iters_SPIDER = num_iters_base_offline;
    print_freq_SPIDER = print_freq_offline;
    num_iters_SVRG = num_iters_base_offline;
    print_freq_SVRG = print_freq_offline;
end
% num_iters_FW = ceil(num_iters_base * batch_size * 2 / data_size);
num_iters_FW = num_iters_base_batch;

W0 = cell(4, 1);
W0{1} = zeros(dimension, model_dim_hidden);  % d-by-m
W0{2} = zeros(model_dim_hidden, 1);  % m-by-1
W0{3} = zeros(model_dim_hidden, num_classes);  % m-by-C
W0{4} = zeros(num_classes, 1);  % C-by-1

loss_handle = @loss_fn;
grad_handle = @grad_fn;
lmo_handle = @(V)lmo_fn(V, model_radius_W, model_radius_b);
gap_handle = @(W, X, y)gap_fn(W, X, y, model_radius_W, model_radius_b);
evaluate_accuracy_handle = @evaluate_accuracy_fn;

obj_values_cell = cell(length(selected_methods), 1);
if ~IS_ADVERSARIAL
    % random_seed = rng('shuffle');
end
for method_idx = 1 : length(selected_methods)
    if ~IS_ADVERSARIAL
        % rng(random_seed);
        % rng(random_seed, 'simdTwister');
    end
    curr_method = selected_methods{method_idx};
    if strcmp(curr_method, 'ORGFW') == true  % ORGFW
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_ORGFW), ', \eta exp=', num2str(eta_exp_ORGFW), ', \rho coef=', num2str(rho_coef_ORGFW), ', \rho exp=', num2str(rho_exp_ORGFW)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = ORGFW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_ORGFW, batch_size, eta_coef_ORGFW, eta_exp_ORGFW, rho_coef_ORGFW, rho_exp_ORGFW, loss_handle, grad_handle, lmo_handle, gap_handle, evaluate_accuracy_handle, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution);
    elseif strcmp(curr_method, 'OSFW') == true  % OSFW
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_OSFW), ', \eta exp=', num2str(eta_exp_OSFW), ', \rho coef=', num2str(rho_coef_OSFW), ', \rho exp=', num2str(rho_exp_OSFW)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = OSFW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_OSFW, batch_size, eta_coef_OSFW, eta_exp_OSFW, rho_coef_OSFW, rho_exp_OSFW, loss_handle, grad_handle, lmo_handle, gap_handle, evaluate_accuracy_handle, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution);
    elseif strcmp(curr_method, 'FW') == true  % Full FW
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_FW), ', \eta exp=', num2str(eta_exp_FW)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = FW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_FW, eta_coef_FW, eta_exp_FW, loss_handle, grad_handle, lmo_handle, gap_handle, evaluate_accuracy_handle, print_freq_FW, IS_ADVERSARIAL);
    elseif strcmp(curr_method, 'OFW') == true  % OFW
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_OFW), ', \eta exp=', num2str(eta_exp_OFW)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = OFW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_OFW, batch_size, eta_coef_OFW, eta_exp_OFW, loss_handle, grad_handle, lmo_handle, gap_handle, evaluate_accuracy_handle, print_freq_OAW, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution);
    elseif strcmp(curr_method, 'ROFW') == true  % ROFW
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_ROFW), ', \eta exp=', num2str(eta_exp_ROFW), ', reg coef=', num2str(reg_coef_ROFW)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = ROFW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_ROFW, batch_size, eta_coef_ROFW, eta_exp_ROFW, reg_coef_ROFW, loss_handle, grad_handle, lmo_handle, gap_handle, evaluate_accuracy_handle, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution);
    elseif strcmp(curr_method, 'OAW') == true  % OAW
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_OAW), ', \eta exp=', num2str(eta_exp_OAW)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = OAW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_OAW, batch_size, eta_coef_OAW, eta_exp_OAW, loss_handle, grad_handle, lmo_handle, gap_handle, evaluate_accuracy_handle, print_freq_OAW, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution);
    elseif strcmp(curr_method, 'MFW') == true  % MFW
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_MFW), ', \eta exp=', num2str(eta_exp_MFW), ', \rho coef=', num2str(rho_coef_MFW), ', \rho exp=', num2str(rho_exp_MFW)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = MFW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_MFW, batch_size, sub_batch_size, eta_coef_MFW, eta_exp_MFW, rho_coef_MFW, rho_exp_MFW, reg_coef_MFW, loss_handle, grad_handle, lmo_handle, gap_handle, evaluate_accuracy_handle, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution);
    elseif strcmp(curr_method, 'MORGFW') == true  % MORGFW
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_MORGFW), ', \eta exp=', num2str(eta_exp_MORGFW), ', \rho coef=', num2str(rho_coef_MORGFW), ', \rho exp=', num2str(rho_exp_MORGFW), ', reg coef=', num2str(reg_coef_MORGFW)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = MORGFW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_MORGFW, batch_size, sub_batch_size, eta_coef_MORGFW, eta_exp_MORGFW, rho_coef_MORGFW, rho_exp_MORGFW, reg_coef_MORGFW, loss_handle, grad_handle, lmo_handle, gap_handle, evaluate_accuracy_handle, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution);
    elseif strcmp(curr_method, 'SPIDER_FW') == true  % SPIDER
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_SPIDER), ', \eta exp=', num2str(eta_exp_SPIDER), ', epoch_length=', num2str(epoch_length_SPIDER), ', batch_size=', num2str(batch_size_SPIDER)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = SPIDER_FW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_SPIDER, batch_size_SPIDER, eta_coef_SPIDER, eta_exp_SPIDER, epoch_length_SPIDER, loss_handle, grad_handle, lmo_handle, gap_handle, evaluate_accuracy_handle, print_freq_SPIDER, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution);
    elseif strcmp(curr_method, 'SVRG_FW') == true  % SVRG
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_SVRG), ', \eta exp=', num2str(eta_exp_SVRG), ', epoch_length=', num2str(epoch_length_SVRG), ', batch_size=', num2str(batch_size_SVRG)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = SVRG_FW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_SVRG, batch_size_SVRG, eta_coef_SVRG, eta_exp_SVRG, epoch_length_SVRG, loss_handle, grad_handle, lmo_handle, gap_handle, evaluate_accuracy_handle, print_freq_SVRG, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution);
    else
        error('method name must be FW, OSFW, OFW, OAW, ROFW, ORGFW, MFW, MORGFW, SPIDER_FW, or SVRG_FW');
    end

    % plot curves
    if IS_OPTIMAL_VALUE_AVAILABLE
        if IS_SFO_FIXED
            semilogy(obj_values(:, 3), (obj_values(:, 4) - optimal_value) ./ (obj_values(1, 4) - optimal_value), 'DisplayName', curr_label); hold on;
            xlabel('time (s)');
            ylabel('loss value');
            legend('show');
        else
            if IS_CALCULATING_REGRET
                plot(obj_values(:, 1), cumsum(obj_values(:, 4)), 'DisplayName', curr_label); hold on;
                xlabel('#iterations');
                ylabel('regret');
                legend('show', 'Location', 'northwest');
            else
                semilogy(obj_values(:, 3), (obj_values(:, 4) - optimal_value) ./ (obj_values(1, 4) - optimal_value), 'DisplayName', curr_label); hold on;
                xlabel('time (s)');
                ylabel('loss value');
                legend('show');
            end
        end
    else
        if IS_SFO_FIXED
            plot(obj_values(:, 3), obj_values(:, 4), 'DisplayName', curr_label); hold on;
            xlabel('time (s)');
            ylabel('loss value');
            legend('show');
        else
            plot(obj_values(:, 1), obj_values(:, 4), 'DisplayName', curr_label); hold on;
            xlabel('#iterations');
            ylabel('loss value');
            legend('show');
        end
    end
    obj_values_cell{method_idx} = obj_values;

    % training accuracy
    training_accuracy = evaluate_accuracy_fn(solution, Xtrain, ytrain);
    test_accuracy = evaluate_accuracy_fn(solution, Xtest, ytest);
    fprintf('training accuracy: %f, test accuracy: %f\n', training_accuracy, test_accuracy);
end

% save the experimental results
output_file_name = [DATA_ROOT, 'results_', dataset, '_NN_auto_save.mat'];
save(output_file_name, 'selected_methods', 'obj_values_cell');

grid on;


%% Section 5: Definitions of loss function, gradient, and linear optimization oracle

% objective function: multiclass logistic regression, see section 5 of https://arxiv.org/pdf/1902.06332
% f(W) = - \sum_{i=1}^N \sum_{c=1}^C 1{y_i = c} log( (exp(w_c^T x_i)) /
% (\sum_{j=1}^C exp(w_j^T x_i))), s.t. ||W||_1 <= 1

% W1: d-by-m
% b1: m-by-1
% W2: m-by-C
% b2: C-by-1

function loss = loss_fn(W, X, y)
    data_size = size(X, 2);
    W1 = W{1};
    b1 = W{2};
    W2 = W{3};
    b2 = W{4};
    sigmoid = 1.0 ./ (1.0 + exp(-(W1' * X + b1)));  % m-by-n
    P = W2' * sigmoid + b2;  % C-by-n

    tmp_exp = exp(P);
    tmp_numerator = zeros(1, data_size);  % 1-by-n
    for i = 1 : data_size
        tmp_numerator(i) = tmp_exp(y(i), i);
    end
    loss = - mean(log(tmp_numerator ./ sum(tmp_exp, 1)));  % average loss
end

function grad = grad_fn(W, X, y)
    data_size = size(X, 2);
    W1 = W{1};
    b1 = W{2};
    W2 = W{3};
    b2 = W{4};
    grad = cell(size(W));

    % forward
    U = W1' * X + b1;
    sigmoid = 1.0 ./ (1.0 + exp(-U));  % m-by-n
    P = W2' * sigmoid + b2;  % C-by-n
    % backward
    tmp_exp = exp(P);  % C-by-n
    tmp_denominator = sum(tmp_exp, 1);
    tmp_exp = tmp_exp ./ tmp_denominator;
    for i = 1 : data_size
        tmp_exp(y(i), i) = tmp_exp(y(i), i) - 1;
    end
    d_P = tmp_exp;  % C-by-n
    d_W2 = sigmoid * d_P' ./ data_size; % m-by-C
    d_b2 = mean(d_P, 2);  % C-by-1
    d_U = (W2 * d_P) .* (sigmoid .* (1 - sigmoid));  % m-by-n
    d_W1 = X * d_U' ./ data_size;  % d-by-m
    d_b1 = mean(d_U, 2);  % m-by-1

    grad{1} = d_W1;
    grad{2} = d_b1;
    grad{3} = d_W2;
    grad{4} = d_b2;
end

function res = lmo_fn(V, radius_W, radius_b)  % lmo for l_1 constranit || W_j ||_1 <= radius
    res = cell(size(V));
    radius_arr = [radius_W; radius_b; radius_W; radius_b];
    for i = 1 : 4
        tmpV = V{i};
        [num_rows, num_cols] = size(tmpV);
        [~, rows] = max(abs(tmpV), [], 1);
        cols = 1:num_cols;
        nz_values = - radius_arr(i) * sign(tmpV(rows + (cols - 1) * num_rows));
        res{i} = sparse(rows, cols, nz_values, num_rows, num_cols, num_cols);
    end
end

function res = projection_fn(V, radius_W, radius_b)  % lmo for l_1 constranit || W_j ||_1 <= radius
    grad = cell(size(V));
    radius_arr = [radius_W; radius_b; radius_W; radius_b];
    for i = 1 : 4
        tmpV = V{i};
        [num_rows, num_cols] = size(tmpV);
        tmpres = zeros(size(tmpV));
        for j = 1 : num_cols
            tmpres(:, j) = ProjectOntoL1Ball(tmpV(:, j), radius_arr(i));
        end
        res{i} = tmpres;
    end
end

function fw_gap = gap_fn(W, X, y, radius_W, radius_b)
    grad = grad_fn(W, X, y);
    V = lmo_fn(grad, radius_W, radius_b);
    fw_gap = 0;
    for i = 1 : 4
        fw_gap = fw_gap + sum(sum(grad{i} .* (W{i} - V{i})));
    end
end

function accuracy = evaluate_accuracy_fn(W, X, y)
    data_size = size(X, 2);
    W1 = W{1};
    b1 = W{2};
    W2 = W{3};
    b2 = W{4};
    sigmoid = 1.0 ./ (1.0 + exp(-(W1' * X + b1)));  % m-by-n
    P = W2' * sigmoid + b2;  % C-by-n
    [~, predict_classes] = max(P);
    accuracy = sum(y' == predict_classes) / data_size;
end
