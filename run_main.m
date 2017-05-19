%% RUN_MAIN.m
%
%  AUTHORS: Salah Nouri
%           Marco Zugliani

clear;
close;

%% Parameters

% type of features to use
% possible values: 'rgb', 'depth','lbp-top'
features_type = 'rgb';

% whether to make only the test step (logical value)
% useful if the features have already been extracted
% we provide already extracted features in Code\3.Results\c.features\
% feature extraction can last up to ~2 hours for lbp-top
test_only     = false;

% the c parameters for the SVM (scalar value)
c             = [];

% the gamma parameters for the SVM (scalar value)
gamma         = [];

tic;
%% MAIN
disp('--- EXECUTION STARTED ---');
main(features_type, test_only, c, gamma);
disp('--- END OF EXECUTION ---');
q = toc;

disp(['Time of execution ' num2str(q) ' [s]']);

warning('off')
clear all;
close all;
warning('on')
