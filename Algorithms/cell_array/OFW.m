function [solution, obj_values] = OFW(W1, Xtrain, ytrain, Xtest, ytest, num_iters, batch_size, eta_coef, eta_exp, loss_fn, grad_fn, lmo_fn, gap_fn, evaluate_accuracy_fn, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution)
    % initialization
    overhead_time = 0.0;

    [~, data_size] = size(Xtrain);
    W = W1;  % copy on write

    obj_values = zeros(fix(num_iters / print_freq) + 1, 7);
    % each row records [#iters, #SFO, running_time, loss_value, FW_gap];
    if IS_CALCULATING_REGRET
        obj_values(1, :) = [0, 0, 0.0, 0.0, 0.0, 0.0, 0.0];
    else
        obj_values(1, :) = [0, 0, 0.0, loss_fn(W, Xtrain, ytrain), gap_fn(W, Xtrain, ytrain), evaluate_accuracy_fn(W, Xtrain, ytrain), evaluate_accuracy_fn(W, Xtest, ytest)];
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
        grad_estimate = cell(size(W1));
        for tmpi = 1 : length(grad_estimate)
            grad_estimate{tmpi} = zeros(size(W1{tmpi}));  % gradient estimator
        end
        for tmp_t = 1 : t
            for tmpi = 1 : length(grad_estimate)
                tmp_grad = grad_fn(W, observed_X_cell_arr{tmp_t}, observed_y_cell_arr{tmp_t});
                grad_estimate{tmpi} = grad_estimate{tmpi} + tmp_grad{tmpi};
            end
        end
        for tmpi = 1 : length(grad_estimate)
            grad_estimate{tmpi} = grad_estimate{tmpi} ./ t;
        end
        % LMO
        V = lmo_fn(grad_estimate);
        % update W
        W_old = W;
        for tmpi = 1 : length(W)
            W{tmpi} = (1 - eta) * W{tmpi} + eta * V{tmpi};
        end

        % evaluate loss function value and FW gap
        if mod(t, print_freq) == 0
            t_current = toc(t_start);
            running_time = t_current - overhead_time;
            if IS_CALCULATING_REGRET
                curr_loss = loss_fn(W_old, observed_X_cell_arr{t}, observed_y_cell_arr{t}) - loss_fn(optimal_solution, observed_X_cell_arr{t}, observed_y_cell_arr{t});
                curr_gap = gap_fn(W_old, observed_X_cell_arr{t}, observed_y_cell_arr{t});
                curr_training_accu = evaluate_accuracy_fn(W, Xtmp, ytmp);
                curr_test_accu = evaluate_accuracy_fn(W, Xtest, ytest);
            else
                curr_loss = loss_fn(W, Xtrain, ytrain);
                curr_gap = gap_fn(W, Xtrain, ytrain);
                % curr_gap = 0.0;
                curr_training_accu = evaluate_accuracy_fn(W, Xtrain, ytrain);
                curr_test_accu = evaluate_accuracy_fn(W, Xtest, ytest);
            end
            obj_values(fix(t / print_freq) + 1, :) = [t, (t+1)*t/2 * batch_size, running_time, curr_loss, curr_gap, curr_training_accu, curr_test_accu];
            overhead_time = overhead_time + toc(t_start) - t_current;
        end
    end
    solution = W;
end
