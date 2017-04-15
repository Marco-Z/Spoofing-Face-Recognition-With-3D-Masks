clear all; clc;

%% load data
folder = 'E:\Marco\MDS project\3DMAD\out\';

load([folder,'train_data']);
load([folder,'train_groups']);
load([folder,'test_data']);
load([folder,'test_groups']);

C     = 2048;
gamma = 0.0078125;

%% convert data to double (libsvm takes double arrays as input)
trg = train_groups;
train_groups = zeros(length(train_groups),1);
teg = test_groups;
test_groups = zeros(length(test_groups),1);

for i = 1:(length(train_groups))
    if trg(i,:) == 'fake'
        train_groups(i) = 0;
    else
        train_groups(i) = 1;
    end
end

for i = 1:(length(test_groups))
    if teg(i,:) == 'fake'
        test_groups(i) = 0;
    else
        test_groups(i) = 1;
    end
end

train_features = double(train_features);
test_features = double(test_features);

%%

model = svmtrain(train_groups, train_features,sprintf('-c %f -g %f -b 1', C, gamma));
%%

[out_groups, accuracy, probs] = svmpredict(test_groups, test_features, model, '-b 1');

% accuracy = sum(out_groups == test_groups)/numel(test_groups);

%%

[X,Y,t] = perfcurve(test_groups,probs(:,2),1);
figure;
plot(X,Y)
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification with LBP on RGB data')







