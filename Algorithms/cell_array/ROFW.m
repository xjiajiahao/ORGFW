function [solution, obj_values] = ROFW(W1, Xtrain, ytrain, Xtest, ytest, num_iters, batch_size, eta_coef, eta_exp, reg_coef, loss_fn, grad_fn, lmo_fn, gap_fn, evaluate_accuracy_fn, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution)
    % initialization
    overhead_time = 0.0;

    [~, data_size] = size(Xtrain);
    W = W1;  % copy on write
    observed_grad_sum = cell(size(W1));
    grad_estimate = cell(size(W1));
    for tmpi = 1 : length(grad_estimate)
        observed_grad_sum{tmpi} = zeros(size(W1{tmpi}));
    end

    obj_values = zeros(fix(num_iters / print_freq) + 1, 7);
    % each row records [#iters, #SFO, running_time, loss_value, FW_gap];
    if IS_CALCULATING_REGRET
        obj_values(1, :) = [0, 0, 0.0, 0.0, 0.0, 0.0, 0.0];
    else
        obj_values(1, :) = [0, 0, 0.0, loss_fn(W, Xtrain, ytrain), gap_fn(W, Xtrain, ytrain), evaluate_accuracy_fn(W, Xtrain, ytrain), evaluate_accuracy_fn(W, Xtest, ytest)];
    end

    reg_coef = reg_coef / num_iters^(3/4);
    % @NOTE see Hazan's OLO book, Theorem 7.2

    t_start = tic;  % timing

    for t = 1 : num_iters
        eta = min(eta_coef / (t + 1)^eta_exp, 1.0);
        % sample an index
        if IS_ADVERSARIAL
            idx = [(t-1) * batch_size + 1: t * batch_size];
        else
            idx = randi([1, data_size], [batch_size, 1]);
        end
        W_old = W;
        % update grad_estimate
        Xtmp = Xtrain(:, idx);
        ytmp = ytrain(idx);
        stoch_grad = grad_fn(W, Xtmp, ytmp);
        for tmpi = 1 : length(observed_grad_sum)
            observed_grad_sum{tmpi} = observed_grad_sum{tmpi} + stoch_grad{tmpi};
            grad_estimate{tmpi} = observed_grad_sum{tmpi} + (W{tmpi} - W1{tmpi}) * 2 / reg_coef;
        end
        % LMO
        V = lmo_fn(grad_estimate);
        % update W
        for tmpi = 1 : length(W)
            W{tmpi} = (1 - eta) * W{tmpi} + eta * V{tmpi};
        end

        % evaluate loss function value and FW gap
        if mod(t, print_freq) == 0
            t_current = toc(t_start);
            running_time = t_current - overhead_time;
            if IS_CALCULATING_REGRET
                curr_loss = loss_fn(W_old, Xtmp, ytmp) - loss_fn(optimal_solution, Xtmp, ytmp);
                curr_gap = gap_fn(W_old, Xtmp, ytmp);
                curr_training_accu = evaluate_accuracy_fn(W, Xtmp, ytmp);
                curr_test_accu = evaluate_accuracy_fn(W, Xtest, ytest);
            else
                curr_loss = loss_fn(W, Xtrain, ytrain);
                curr_gap = gap_fn(W, Xtrain, ytrain);
                % curr_gap = 0.0;
                curr_training_accu = evaluate_accuracy_fn(W, Xtrain, ytrain);
                curr_test_accu = evaluate_accuracy_fn(W, Xtest, ytest);
            end
            obj_values(fix(t / print_freq) + 1, :) = [t, t * batch_size, running_time, curr_loss, curr_gap, curr_training_accu, curr_test_accu];
            overhead_time = overhead_time + toc(t_start) - t_current;
        end
    end
    solution = W;
end
