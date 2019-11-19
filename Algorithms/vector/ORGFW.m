function [solution, obj_values] = ORGFW(W1, Xtrain, ytrain, Xtest, ytest, num_iters, batch_size, eta_coef, eta_exp, rho_coef, rho_exp, loss_fn, grad_fn, lmo_fn, gap_fn, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution)
    if IS_ADVERSARIAL
        error('Go home, OSFW only applies to the stochastic setting.');
    end
    % initialization
    overhead_time = 0.0;

    [~, data_size] = size(Xtrain);
    W = W1;  % copy on write

    obj_values = zeros(fix(num_iters / print_freq) + 1, 5);
    % each row records [#iters, #SFO, running_time, loss_value, FW_gap];
    if IS_CALCULATING_REGRET
        obj_values(1, :) = [0, 0, 0.0, 0.0, 0.0];
    else
        obj_values(1, :) = [0, 0, 0.0, loss_fn(W, Xtrain, ytrain), gap_fn(W, Xtrain, ytrain)];
    end


    t_start = tic;  % timing
    % rng(1);

    for t = 1 : num_iters
        eta = min(eta_coef / (t + 1)^eta_exp, 1.0);
        rho = min(rho_coef / (t + 1)^rho_exp, 1.0);
        % sample an index
        idx = randi([1, data_size], [batch_size, 1]);
        Xtmp = Xtrain(:, idx);
        ytmp = ytrain(idx);
        % update grad_estimate
        if t == 1
            grad_estimate = grad_fn(W, Xtmp, ytmp);
        else
            stoch_grad = grad_fn(W, Xtmp, ytmp);
            stoch_grad_old = grad_fn(W_old, Xtmp, ytmp);
            grad_estimate = (1 - rho) * (grad_estimate - stoch_grad_old) + stoch_grad;
        end
        % LMO
        V = lmo_fn(grad_estimate);
        % update W
        W_old = W;
        W = (1 - eta) * W + eta * V;

        % evaluate loss function value and FW gap
        if mod(t, print_freq) == 0
            t_current = toc(t_start);
            running_time = t_current - overhead_time;
            if IS_CALCULATING_REGRET
                curr_loss = loss_fn(W_old, Xtmp, ytmp) - loss_fn(optimal_solution, Xtmp, ytmp);
                curr_gap = gap_fn(W, Xtmp, ytmp);
            else
                curr_loss = loss_fn(W, Xtrain, ytrain);
                curr_gap = gap_fn(W, Xtrain, ytrain);
                % curr_gap = 0.0;
            end
            obj_values(fix(t / print_freq) + 1, :) = [t, t * batch_size * 2, running_time, curr_loss, curr_gap];
            overhead_time = overhead_time + toc(t_start) - t_current;
        end
    end
    solution = W;
end
