function [solution, obj_values] = OAW(W1, Xtrain, ytrain, Xtest, ytest, num_iters, batch_size, eta_coef, eta_exp, loss_fn, grad_fn, lmo_fn, gap_fn, evaluate_accuracy_fn, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution)
    % initialization
    % @NOTE W1 has to be 0
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

    active_cell_arr = {};
    weight_cell_arr = {};
    num_FW_steps_taken = 0;

    t_start = tic;  % timing
    % rng(1);

    for t = 1 : num_iters
        % sample an index
        if IS_ADVERSARIAL
            idx = [(t-1) * batch_size + 1: t * batch_size];
        else
            idx = randi([1, data_size], [batch_size, 1]);
        end
        W_old = W;
        % store data
        observed_X_cell_arr{t} = Xtrain(:, idx);
        observed_y_cell_arr{t} = ytrain(idx);
        % update grad_estimate
        grad_estimate = cell(size(W1));
        for tmp_layer = 1 : length(grad_estimate)
            grad_estimate{tmp_layer} = zeros(size(W1{tmp_layer}));  % gradient estimator
        end
        for tmp_t = 1 : t
            for tmp_layer = 1 : length(grad_estimate)
                tmp_grad = grad_fn(W, observed_X_cell_arr{tmp_t}, observed_y_cell_arr{tmp_t});
                grad_estimate{tmp_layer} = grad_estimate{tmp_layer} + tmp_grad{tmp_layer};
            end
        end
        for tmp_layer = 1 : length(grad_estimate)
            grad_estimate{tmp_layer} = grad_estimate{tmp_layer} ./ t;
        end
        % LMO
        V = lmo_fn(grad_estimate);
        % compute the away direction
        if ~isempty(active_cell_arr)
            max_dot_prod_val = -inf;
            max_dot_prod_idx = 0;
            for tmp_i = 1 : length(active_cell_arr)
                curr_dot_prod = 0.0;
                tmp_active_vertex = active_cell_arr{tmp_i};
                for tmp_layer = 1 : length(grad_estimate)
                    curr_dot_prod = curr_dot_prod + sum(sum(tmp_active_vertex{tmp_layer} .* grad_estimate{tmp_layer}));
                end
                if curr_dot_prod > max_dot_prod_val
                    max_dot_prod_val = curr_dot_prod;
                    max_dot_prod_idx = tmp_i;
                end
            end
        end
        % determine whether we take a FW step or away step or drop step
        dot_prod_V_grad = 0.0;
        dot_prod_W_grad = 0.0;
        for tmp_layer = 1 : length(grad_estimate)
            dot_prod_V_grad = dot_prod_V_grad + sum(sum(V{tmp_layer} .* grad_estimate{tmp_layer}));
            dot_prod_W_grad = dot_prod_W_grad + sum(sum(W{tmp_layer} .* grad_estimate{tmp_layer}));
        end
        if isempty(active_cell_arr) || dot_prod_V_grad + max_dot_prod_val <= 2 * dot_prod_W_grad
            % FW step
            num_FW_steps_taken = num_FW_steps_taken + 1;
            eta = min(eta_coef / (num_FW_steps_taken + 1)^eta_exp, 1.0);
            for tmp_layer = 1 : length(grad_estimate)
                W{tmp_layer} = (1 - eta) * W{tmp_layer} + eta * V{tmp_layer};
            end
            % fw_op(active_cell_arr, weight_cell_arr, eta, V);
            IS_V_ACTIVE = false;
            idx_V = 0;
            for tmp_i = 1 : length(active_cell_arr)
                tmp_active_vertex = active_cell_arr{tmp_i};
                is_equal_flag = true;
                for tmp_layer = 1 : length(grad_estimate)
                    if isequal(V{tmp_layer}, tmp_active_vertex{tmp_layer})
                        is_equal_flag = false;
                        break;
                    end
                end
                if is_equal_flag
                    IS_V_ACTIVE = true;
                    idx_V = tmp_i;
                    break;
                end
            end
            for tmp_i = 1 : length(weight_cell_arr)
                weight_cell_arr{tmp_i} = weight_cell_arr{tmp_i} * (1 - eta);
            end
            if IS_V_ACTIVE
                weight_cell_arr{idx_V} = weight_cell_arr{idx_V} + eta;
            else  % insert new vertex
                active_cell_arr{end+1} = V;
                weight_cell_arr{end+1} = eta;
            end
        else
            % away or drop step
            eta_max = 1.0/(1 - weight_cell_arr{max_dot_prod_idx}) - 1;
            if eta_max >= eta  % away step
                eta = eta_coef / (num_FW_steps_taken + 1)^eta_exp;
                for tmp_layer = 1 : length(grad_estimate)
                    W{tmp_layer} = (1 + eta) * W{tmp_layer} - eta * active_cell_arr{max_dot_prod_idx}{tmp_layer};
                end
                % away_op(active_cell_arr, weight_cell_arr, eta, max_dot_prod_idx);
                for tmp_i = 1 : length(weight_cell_arr)
                    weight_cell_arr{tmp_i} = weight_cell_arr{tmp_i} * (1 + eta);
                end
                weight_cell_arr{max_dot_prod_idx} = weight_cell_arr{max_dot_prod_idx} - eta;
            else  % drop step
                eta = eta_max;
                for tmp_layer = 1 : length(grad_estimate)
                    W{tmp_layer} = (1 + eta) * W{tmp_layer} - eta * active_cell_arr{max_dot_prod_idx}{tmp_layer};
                end
                % drop_op(active_cell_arr, weight_cell_arr, eta, max_dot_prod_idx);
                weight_cell_arr(max_dot_prod_idx) = [];
                active_cell_arr(max_dot_prod_idx) = [];
                for tmp_i = 1 : length(weight_cell_arr)
                    weight_cell_arr{tmp_i} = weight_cell_arr{tmp_i} * (1 + eta);
                end
            end
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
