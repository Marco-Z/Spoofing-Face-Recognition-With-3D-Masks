clear all; clc;

%% load data
folder = 'E:\Marco\MDS project\3DMAD\out\';

load([folder,'train_data']);
load([folder,'train_groups']);
load([folder,'dev_data']);
load([folder,'dev_groups']);

%% convert data to double (libsvm takes double arrays as input)
trg = train_groups;
train_groups = zeros(length(train_groups),1);
teg = dev_groups;
dev_groups = zeros(length(dev_groups),1);

for i = 1:(length(train_groups))
    if trg(i,:) == 'fake'
        train_groups(i) = 0;
    else
        train_groups(i) = 1;
    end
end

for i = 1:(length(dev_groups))
    if teg(i,:) == 'fake'
        dev_groups(i) = 0;
    else
        dev_groups(i) = 1;
    end
end

train_features = double(train_features);
dev_features = double(dev_features);
% d_train_features = double(d_train_features);
% d_test_features = double(d_test_features);

%%

%grid of parameters
folds = 5; 
[C,gamma] = meshgrid(-5:2:15, -15:2:3); 
%# grid search, and cross-validation 
cv_acc = zeros(numel(C),1); 
d= 2;
for i=1:numel(C)   
    cv_acc(i) = svmtrain(train_groups, train_features,sprintf('-c %f -g %f -v %d -t %d', 2^C(i), 2^gamma(i), folds,d));
end
%# pair (C,gamma) with best accuracy
[~,idx] = max(cv_acc); 
%# contour plot of paramter selection 
contour(C, gamma, reshape(cv_acc,size(C))), colorbar
hold on;
text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...  
    'HorizontalAlign','left', 'VerticalAlign','top') 
hold off 
xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy') 
%# now you can train you model using best_C and best_gamma
best_C = 2^C(idx); best_gamma = 2^gamma(idx); %# ...