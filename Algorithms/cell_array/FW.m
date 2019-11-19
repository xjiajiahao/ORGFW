function [solution, obj_values] = FW(W1, Xtrain, ytrain, Xtest, ytest, num_iters, eta_coef, eta_exp, loss_fn, grad_fn, lmo_fn, gap_fn, evaluate_accuracy_fn, print_freq, IS_ADVERSARIAL)
    % initialization
    overhead_time = 0.0;

    [~, data_size] = size(Xtrain);
    W = W1;  % copy on write

    obj_values = zeros(fix(num_iters / print_freq) + 1, 7);
    % each row records [#iters, #SFO, running_time, loss_value, FW_gap];
    obj_values(1, :) = [0, 0, 0.0, loss_fn(W, Xtrain, ytrain), gap_fn(W, Xtrain, ytrain), evaluate_accuracy_fn(W, Xtrain, ytrain), evaluate_accuracy_fn(W, Xtest, ytest)];

    t_start = tic;  % timing

    grad_estimate = grad_fn(W, Xtrain, ytrain);

    for t = 1 : num_iters
        eta = min(eta_coef / (t + 1)^eta_exp, 1.0);
        % LMO
        V = lmo_fn(grad_estimate);
        for tmpi = 1 : length(W)
            W{tmpi} = (1 - eta) * W{tmpi} + eta * V{tmpi};
        end
        % update grad_estimate
        grad_estimate = grad_fn(W, Xtrain, ytrain);

        % evaluate loss function value and FW gap
        if mod(t, print_freq) == 0
            t_current = toc(t_start);
            running_time = t_current - overhead_time;
            curr_loss = loss_fn(W, Xtrain, ytrain);
            curr_gap = gap_fn(W, Xtrain, ytrain);
            % curr_gap = 0.0;   % @NOTE just for finding the optimal value
            curr_training_accu = evaluate_accuracy_fn(W, Xtrain, ytrain);
            curr_test_accu = evaluate_accuracy_fn(W, Xtest, ytest);
            obj_values(fix(t / print_freq) + 1, :) = [t, t * data_size, running_time, curr_loss, curr_gap, curr_training_accu, curr_test_accu];
            overhead_time = overhead_time + toc(t_start) - t_current;
        end
    end
    solution = W;
end
