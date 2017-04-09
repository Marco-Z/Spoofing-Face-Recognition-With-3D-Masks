clear all; clc;

folder = 'E:\Marco\MDS project\3DMAD\';
% folder/
% |-- real  -> all the real faces here
% |-- fake  -> all the fake faces here
% |-- train
% |   |-- real
% |   `-- fake
% |-- dev
% |   |-- real
% |   `-- fake
% |-- test
% |   |-- real
% |	  `-- fake
% `-- out
%     |-- train
%     |   |-- real
%     |   `-- fake
%     |-- dev
%     |   |-- real
%     |   `-- fake
%     `-- test
%         |-- real
%         `-- fake

%crete output folders
warning('off');

mkdir([folder,'train\']);
mkdir([folder,'dev\']);
mkdir([folder,'test\']);
mkdir([folder,'out\']);

mkdir([folder,'train\fake\']);
mkdir([folder,'train\real\']);
mkdir([folder,'dev\fake\']);
mkdir([folder,'dev\real\']);
mkdir([folder,'test\fake\']);
mkdir([folder,'test\real\']);

mkdir([folder,'out\train\']);
mkdir([folder,'out\dev\']);
mkdir([folder,'out\test\']);

mkdir([folder,'out\train\fake\']);
mkdir([folder,'out\train\real\']);
mkdir([folder,'out\dev\fake\']);
mkdir([folder,'out\dev\real\']);
mkdir([folder,'out\test\fake\']);
mkdir([folder,'out\test\real\']);

warning('on');

%% split real and fake into train, dev and test sets

train = [29 58];
dev = [28 56];
test = [28 56];
% total = [85 170]

f = ls([folder,'fake\','*.hdf5']);
r = ls([folder,'real\','*.hdf5']);

fn = 1:size(f,1);
rn = 1:size(r,1);

% randomize
% f = f(fn(randperm(size(f,1))),:);
% r = r(rn(randperm(size(r,1))),:);

train_files_fake = (f(1:train(1),:));
dev_files_fake   = (f(train(1)+1:train(1)+dev(1),:));
test_files_fake  = (f(train(1)+dev(1)+1:end,:));
train_files_real = (r(1:train(2),:));
dev_files_real   = (r(train(2)+1:train(2)+dev(2),:));
test_files_real  = (r(train(2)+dev(2)+1:end,:));

%% move files in specific folder

batch_move(folder,'fake\','train\fake\',train_files_fake, train(1));
batch_move(folder,'fake\','dev\fake\',dev_files_fake,dev(1));
batch_move(folder,'fake\','test\fake\',test_files_fake,test(1));
batch_move(folder,'real\','train\real\',train_files_real,train(2));
batch_move(folder,'real\','dev\real\',dev_files_real,dev(2));
batch_move(folder,'real\','test\real\',test_files_real,test(2));

%% support functions

function [out] = batch_move( base, f1, f2, files, n )
    out = true;
    for i = 1:n
       out = out & movefile([base,f1,files(i,:)], ...
                            [base,f2,files(i,:)]);
    end
end


