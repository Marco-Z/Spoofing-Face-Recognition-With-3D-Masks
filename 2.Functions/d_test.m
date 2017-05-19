%% D_TEST
%  Function used to train and test the SVM classifier with given parameters
%  INPUT:
%         - c       : the cost parameter for the SVM classifier
%         - gamma   : the gamma parameter for the SVM classifier
%  OUTPUT:
%         - accuracy: the accuracy of the classifier on the test set

function [accuracy] = d_test(c,gamma)
disp('-----------------------');
disp('Model training and test');
disp('-----------------------');
%% load data
folder = '..\3.Results\c.features\depth\';
out = '..\3.Results\e.final_results\';

load([folder,'train_data_d']); % train_data_d : matrix that contains the features
% extracted from the trainning samples.
load([folder,'train_groups']); % train_groups : matrix that contains the labels
% of the trainning samples.
load([folder,'test_data_d']);  % like train_data_d
load([folder,'test_groups']);  % like train_groups_d

train_features = train_d_features;
test_features  = test_d_features;

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

%% train
model = svmtrain(train_groups, train_features,sprintf('-q -c %f -g %f -b 1', c, gamma));

%% classify
[out_groups, accuracy, probs] = svmpredict(test_groups, test_features, model, '-b 1');

%% Plot the roc curves
[X,Y,t,~,opt] = perfcurve(test_groups,probs(:,2),1);
hFig = figure(1);
plot(X,Y,'LineWidth',2)
xlabel('False positive rate')
ylabel('True positive rate')
pbaspect([1 1 1])
title('ROC for Classification with LBP on depth data')
warning('off');
mkdir([out 'depth\']);
warning('on');
saveas(1,[out 'depth\ROC'],'png');

end






