function [solution, obj_values] = OFW(W1, Xtrain, ytrain, Xtest, ytest, num_iters, batch_size, eta_coef, eta_exp, loss_fn, grad_fn, lmo_fn, gap_fn, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution)
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

    observed_X_cell_arr = cell(num_iters, 1);
    observed_y_cell_arr = cell(num_iters, 1);

    t_start = tic;  % timing
    % rng(1);

    for t = 1 : num_iters
        eta = min(eta_coef / (t + 1)^eta_exp, 1.0);
        % sample an index
        if IS_ADVERSARIAL
            idx = [(t-1) * batch_size + 1: t * batch_size];
        else
            idx = randi([1, data_size], [batch_size, 1]);
        end
        % store data
        observed_X_cell_arr{t} = Xtrain(:, idx);
        observed_y_cell_arr{t} = ytrain(idx);
        % update grad_estimate
        grad_estimate = zeros(size(W1));
        for tmp_t = 1 : t
            grad_estimate = grad_estimate + grad_fn(W, observed_X_cell_arr{tmp_t}, observed_y_cell_arr{tmp_t});
        end
        grad_estimate = grad_estimate ./ t;
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
                curr_loss = loss_fn(W_old, observed_X_cell_arr{t}, observed_y_cell_arr{t}) - loss_fn(optimal_solution, observed_X_cell_arr{t}, observed_y_cell_arr{t});
                curr_gap = gap_fn(W_old, observed_X_cell_arr{t}, observed_y_cell_arr{t});
                obj_values(fix(t / print_freq) + 1, :) = [t, (t+1)*t/2 * batch_size, running_time, curr_loss, curr_gap];
            else
                curr_loss = loss_fn(W, Xtrain, ytrain);
                curr_gap = gap_fn(W, Xtrain, ytrain);
                % curr_gap = 0.0;
                obj_values(fix(t / print_freq) + 1, :) = [t, (t+1)*t/2 * batch_size, running_time, curr_loss, curr_gap];
            end
            overhead_time = overhead_time + toc(t_start) - t_current;
        end
    end
    solution = W;
end
