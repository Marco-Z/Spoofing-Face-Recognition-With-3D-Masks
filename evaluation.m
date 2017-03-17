clear all; clc;

%% load data

load train_data
load train_data_d
load train_groups
load test_data
load test_data_d
load test_groups

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
d_train_features = double(d_train_features);
d_test_features = double(d_test_features);

% split test set to see accuracy for each class
test_features_fake = test_features(1:15,:);
test_features_real = test_features(16:30,:);
test_features_fake_d = d_test_features(1:15,:);
test_features_real_d = d_test_features(16:30,:);
test_groups_fake = test_groups(1:15,:);
test_groups_real = test_groups(16:30,:);
%% accuracy test

c = 0.1:0.05:10;
g = 0.1:0.1:10;

fake_3d = [];
real_3d = [];
fake_3d_d = [];
real_3d_d = [];
% train model and get accuracy
% '-t 0' for linear kernel
% '-t 2' for radial kernel
for i = 1:length(g)
    func_fake = [];
    func_real = [];
    func_fake_d = [];
    func_real_d = [];
    for j = 1:length(c)
        model = svmtrain(train_groups, train_features, ['-q -t 2 -c ' num2str(c(j)) ' -g ' num2str(g(i))]);

        out_groups_fake = svmpredict(test_groups_fake, test_features_fake, model, '-q');
        out_groups_real = svmpredict(test_groups_real, test_features_real, model, '-q');

        accuracy_fake = sum(out_groups_fake == test_groups_fake)/numel(test_groups_fake);
        accuracy_real = sum(out_groups_real == test_groups_real)/numel(test_groups_real);
        func_fake = [func_fake accuracy_fake];
        func_real = [func_real accuracy_real];


        d_model = svmtrain(train_groups, d_train_features, ['-q -t 2 -c ' num2str(c(j)) ' -g ' num2str(g(i))]);

        out_groups_fake_d = svmpredict(test_groups_fake, test_features_fake_d, model, '-q');
        out_groups_real_d = svmpredict(test_groups_real, test_features_real_d, model, '-q');

        accuracy_fake_d = sum(out_groups_fake_d == test_groups_fake)/numel(test_groups_fake);
        accuracy_real_d = sum(out_groups_real_d == test_groups_real)/numel(test_groups_real);
        func_fake_d = [func_fake_d accuracy_fake_d];
        func_real_d = [func_real_d accuracy_real_d];
    end
    fake_3d = [fake_3d; func_fake];
    real_3d = [real_3d; func_real];
    fake_3d_d = [fake_3d_d; func_fake_d];
    real_3d_d = [real_3d_d; func_real_d];
end
%%
% plot result
[row,col] = size(fake_3d);
[row_d,col_d] = size(fake_3d_d);

surface = zeros(row,col);
surface_d = zeros(row_d,col_d);
for i = 1:row
    for j = 1:col
        if(fake_3d(i,j) == 1 && real_3d(i,j) == 1 )
            surface(i,j) = 1;
        elseif(fake_3d(i,j) > 0.9 && real_3d(i,j) > 0.9 )
            surface(i,j) = 0.5;
        else
            surface(i,j) = 0;
        end
        
        if(fake_3d_d(i,j) == 1 && real_3d_d(i,j) == 1 )
            surface_d(i,j) = 1;
        elseif(fake_3d_d(i,j) > 0.9 && real_3d_d(i,j) > 0.9 )
            surface_d(i,j) = 0.5;
        else
            surface_d(i,j) = 0;
        end
    end
end
            
imshow(surface);
figure;
imshow(surface_d);



