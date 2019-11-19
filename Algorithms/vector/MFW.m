function [solution, obj_values] = MFW(W1, Xtrain, ytrain, Xtest, ytest, num_iters, batch_size, sub_batch_size, eta_coef, eta_exp, rho_coef, rho_exp, reg_coef, loss_fn, grad_fn, lmo_fn, gap_fn, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution)
    if ~IS_ADVERSARIAL
        error('Go home, MFW only applies to the adversarial setting.');
    end
    %% initialization
    overhead_time = 0.0;

    grad_estimate = zeros(size(W1));  % gradient estimator

    obj_values = zeros(fix(num_iters / print_freq) + 1, 5);
    % each row records [#iters, #SFO, running_time, loss_value, FW_gap];
    % obj_values(1, :) = [0, 0, 0.0, 0.0, 0.0];

    num_oracles = ceil(num_iters^(3/2));
    % num_oracles = ceil(num_iters);

    variable_cell_arr = cell(num_oracles, 1);
    grad_cell_arr = cell(num_oracles, 1);

    % reg_coef = 1/sqrt(num_iters);
    reg_coef = reg_coef / sqrt(num_iters);

    t_start = tic;  % timing

    for k = 1 : num_oracles
        grad_cell_arr{k} = grad_estimate;  % zeros
    end

    %% main loop
    for t = 1 : num_iters
        % receive f_t  @NOTE this is illegal because we need to play a variable fist and the receive f_t, but we do so to save memory
        idx = [(t-1) * batch_size + 1: t * batch_size];
        Xtmp = Xtrain(:, idx);
        ytmp = ytrain(idx);

        W = W1;
        grad_estimate = zeros(size(W1));  % gradient estimator
        for k = 1 : num_oracles
            eta = min(eta_coef / (k + 1)^eta_exp, 1.0);
            rho = min(rho_coef / (k + 1)^rho_exp, 1.0);
            %% FTPL prediction
            % sample injected_noise
            % injected_noise = randn(size(W1));
            injected_noise = -0.5 + rand(size(W1));
            V = lmo_fn(grad_cell_arr{k} .* reg_coef + injected_noise);
            % update grad_estimate (pretending that we already have f_t)
            idx = randi([1, batch_size], [sub_batch_size, 1]);
            stoch_grad = grad_fn(W, Xtmp(:, idx), ytmp(idx));
            grad_estimate = (1 - rho) * (grad_estimate) + rho * stoch_grad;
            % feedback the linear objective function to FTPL
            grad_cell_arr{k} = grad_cell_arr{k} + grad_estimate;
            % update W
            W = (1 - eta) * W + eta * V;
        end

        %% evaluate loss function value and FW gap
        if mod(t, print_freq) == 0
            t_current = toc(t_start);
            running_time = t_current - overhead_time;
            curr_loss = loss_fn(W, Xtmp, ytmp) - loss_fn(optimal_solution, Xtmp, ytmp);
            curr_gap = gap_fn(W, Xtmp, ytmp);
            obj_values(fix(t / print_freq) + 1, :) = [t, t * batch_size, running_time, curr_loss, curr_gap];
            overhead_time = overhead_time + toc(t_start) - t_current;
        end
    end
    solution = W;
end
