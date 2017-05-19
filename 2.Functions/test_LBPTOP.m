%% TEST_LBPTOP
%  Function used to train and test the SVM classifier with given parameters
%  INPUT:
%         - c       : the cost parameter for the SVM classifier
%         - gamma   : the gamma parameter for the SVM classifier
%  OUTPUT:
%         - accuracy: the accuracy of the classifier on the test set

function [accuracy] = test_LBPTOP(c,gamma)
disp('-----------------------');
disp('Model training and test');
disp('-----------------------');
%% load data

folder = '..\3.Results\c.features\lbp-top\';
out = '..\3.Results\e.final_results\lbp-top\';

load([folder,'train_data']); % Train_data : matrix that contains the features
% extracted from the trainning samples.
load([folder,'train_groups']);  % Train_groups : matrix contains the lables
% of the training data.
load([folder,'test_data']);   % Matrix like train_data
load([folder,'test_groups']); % Matrix like train_groups

%% convert data to double (libsvm takes double arrays as input)
trg = train_groups;
train_groups = zeros(length(train_groups),1); % initialization
teg = test_groups;
test_groups = zeros(length(test_groups),1); % initialization

% Convert to double the training groups because of library constraints.
for i = 1:(length(train_groups))
    if trg(i,:) == 'fake'
        train_groups(i) = 0;
    else
        train_groups(i) = 1;
    end
end
% Convert to double the testing groups
for i = 1:(length(test_groups))
    if teg(i,:) == 'fake'
        test_groups(i) = 0;
    else
        test_groups(i) = 1;
    end
end
% convert to double
train_features = double(train_features);
test_features = double(test_features);

%% Training the classifier with the best values of 'c' and 'gamma'
model = svmtrain(train_groups, train_features,sprintf('-q -c %f -g %f -b 1', c, gamma));

%%  Testing the classifier
[out_groups, accuracy, probs] = svmpredict(test_groups, test_features, model, '-b 1');

%% Plot the ROC
[X,Y,t,~,opt] = perfcurve(test_groups,probs(:,2),1);
hFig = figure(1);
plot(X,Y,'LineWidth',2)
xlabel('False positive rate')
ylabel('True positive rate')
pbaspect([1 1 1])
title('ROC for Classification with LBP TOP')
warning('off');
mkdir(out);
warning('on');
saveas(1,[out 'ROC'],'png');
end




