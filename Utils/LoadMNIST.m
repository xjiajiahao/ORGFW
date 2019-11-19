% MNIST dataset, http://yann.lecun.com/exdb/mnist/
DATA_ROOT = '../Data/';
Xtrain = loadMNISTImages([DATA_ROOT, 'MNIST/train-images.idx3-ubyte']);
ytrain = loadMNISTLabels([DATA_ROOT, 'MNIST/train-labels.idx1-ubyte']);


Xtest = loadMNISTImages([DATA_ROOT, 'MNIST/t10k-images.idx3-ubyte']);
ytest = loadMNISTLabels([DATA_ROOT, 'MNIST/t10k-labels.idx1-ubyte']);

classes = unique(ytrain);
ytrain = ytrain + 1;
ytest = ytest + 1;

% % standardize features by removing the mean and scaling to unit variance
% data_size = size(Xtrain, 2);
% avg = mean(Xtrain, 2);
% Xtrain = Xtrain - avg;
% stddev = sqrt(sum(sum(Xtrain.^2)) / data_size);
% Xtrain = Xtrain ./ stddev;
% Xtest = (Xtest - avg) ./ stddev;

% % renormalize
% Xtrain = Xtrain ./ sqrt(sum(Xtrain.^2, 1));
% Xtest = Xtest ./ sqrt(sum(Xtest.^2, 1));

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

data_file_name = [DATA_ROOT, 'MNIST_dataset.mat'];
save(data_file_name, 'Xtrain', 'ytrain', 'Xtest', 'ytest');

function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end

function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end
