clear;
DATA_ROOT = '../Data/';

rng(1);  % for reproducibility

% load training data
Xtrain = [];
ytrain = [];
num_data_chunks = 5;
for i = 1 : num_data_chunks
    file_name = [DATA_ROOT, 'CIFAR-10/data_batch_', num2str(i), '.mat'];
    load(file_name);
    Xtrain = [Xtrain, data'];
    ytrain = [ytrain; labels];
end

% load test data
file_name = [DATA_ROOT, 'CIFAR-10/test_batch.mat'];
load(file_name);
Xtest = data';
ytest = labels;

% preprocessing (normalized) x <- (x - min) / (max - min)
Xtrain = double(Xtrain);
min_Xtrain = min(Xtrain);
max_Xtrain = max(Xtrain);
Xtrain = (Xtrain - min_Xtrain) ./ (max_Xtrain - min_Xtrain);

Xtest = double(Xtest);
min_Xtest= min(Xtest);
max_Xtest = max(Xtest);
Xtest= (Xtest - min_Xtest) ./ (max_Xtest - min_Xtest);

% the label of data should be 1 to 10 instead of 0 to 9
ytrain = ytrain + 1;
ytest = ytest + 1;

% reorder
num_classes = length(unique(ytrain));
Xtmp = zeros(size(Xtrain));
ytmp = zeros(size(ytrain));
num_found = 0;
for i = 1 : num_classes
    curr_idx = find(ytrain == i);
    len_curr_idx = length(curr_idx);
    Xtmp(:, num_found + 1 : num_found + len_curr_idx) = Xtrain(:, curr_idx);
    ytmp(num_found + 1 : num_found + len_curr_idx) = ytrain(curr_idx);
    num_found = num_found + len_curr_idx;
end
Xtrain = Xtmp;
ytrain = ytmp;

% save dataset to a file
file_name = [DATA_ROOT, 'CIFAR10_dataset.mat'];
save(file_name, 'Xtrain', 'ytrain', 'Xtest', 'ytest');
