function [solution, obj_values] = FW(W1, Xtrain, ytrain, Xtest, ytest, num_iters, eta_coef, eta_exp, loss_fn, grad_fn, lmo_fn, gap_fn, IS_ADVERSARIAL)
    % initialization
    overhead_time = 0.0;

    [~, data_size] = size(Xtrain);
    W = W1;  % copy on write

    obj_values = zeros(num_iters + 1, 5);
    % each row records [#iters, #SFO, running_time, loss_value, FW_gap];
    obj_values(1, :) = [0, 0, 0.0, loss_fn(W, Xtrain, ytrain), gap_fn(W, Xtrain, ytrain)];

    t_start = tic;  % timing

    grad_estimate = grad_fn(W, Xtrain, ytrain);

    for t = 1 : num_iters
        eta = min(eta_coef / (t + 1)^eta_exp, 1.0);
        % LMO
        V = lmo_fn(grad_estimate);
        W = ( 1 - eta) * W + eta * V;
        % update grad_estimate
        grad_estimate = grad_fn(W, Xtrain, ytrain);

        % evaluate loss function value and FW gap
        t_current = toc(t_start);
        running_time = t_current - overhead_time;
        curr_loss = loss_fn(W, Xtrain, ytrain);
        curr_gap = gap_fn(W, Xtrain, ytrain);
        % curr_gap = 0.0;   % @NOTE just for finding the optimal value
        obj_values(t + 1, :) = [t, t * data_size, running_time, curr_loss, curr_gap];
        overhead_time = overhead_time + toc(t_start) - t_current;
    end
    solution = W;
end
