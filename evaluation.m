%% load data

load train_data
load train_groups
load test_data
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

% split test set to see accuracy for each class
test_features_fake = test_features(1:15,:);
test_features_real = test_features(16:30,:);
test_groups_fake = test_groups(1:15,:);
test_groups_real = test_groups(16:30,:);
%% accuracy test

%build support vector: [1 2 3 ... 1 ... 4 3 2]
%                      [2 3 4 ... 1 ... 3 2 1]
k0 = 1:100;
k = [k0 1 fliplr(k0+1); k0+1 1 fliplr(k0)];

func_fake = [];
func_real = [];
% train model and get accuracy
% '-t 0' for linear kernel
% '-t 2' for radial kernel
for i = 1:length(k)
    model = svmtrain(train_groups, train_features, ['-q -t 2 -w0 ' num2str(k(1,i)) ' -w1 ' num2str(k(2,i))]);

    out_groups_fake = svmpredict(test_groups_fake, test_features_fake, model, '-q');
    out_groups_real = svmpredict(test_groups_real, test_features_real, model, '-q');

    accuracy_fake = sum(out_groups_fake == test_groups_fake)/numel(test_groups_fake);
    accuracy_real = sum(out_groups_real == test_groups_real)/numel(test_groups_real);
    func_fake = [func_fake accuracy_fake];
    func_real = [func_real accuracy_real];
end
% plot result
n = k(1,:) ./ k(2,:);
plot(n,func_fake);
hold on;
plot(n,func_real);
stem(1,1,'k', 'Marker', 'none');
hold off;









