%%extract the data and 
% save them to data.mat and groups.mat

x = ls('*.hdf5');
for i=1:length(x)
    dextr(x(i,:));
end;

%% train svm classifier
load('data.mat');
load('groups.mat');

SVMStruct = svmtrain(data(1:end-1,:),groups(1:end-1,:));

%% classify data

load('data.mat');
load('groups.mat');

group = svmclassify(SVMStruct,data(end,:));

disp(groups(end,:));
disp(group);