%% LANDMARKS
%  Main function
%  INPUT:
%         - features_type: type of features to use
%         - test_only    : whether to make only the test step, useful
%                          if the features have already been extracted
%         - c            : the c parameters for the SVM
%         - gamma        : the gamma parameters for the SVM
%
% proposed values for c and gamma:
%
% features_type | c   | gamma |
% --------------|-----|-------|
% rgb           |  64 |    1  |
% depth         | 256 |  256  |
% lbp_top       |   4 |    1  |

function [] = main( features_type, test_only, c, g )

% Add the Functions folder to the path, in order to use functions inside
addpath(genpath('2.Functions\'));
cd '2.Functions\'

% Create folders in Results folder
warning('off');
mkdir('..\3.Results\a.faces\');
mkdir('..\3.Results\b.normalized\');
mkdir('..\3.Results\c.features\');
mkdir('..\3.Results\d.libsvm_features\');
mkdir('..\3.Results\e.final_results\');
warning('on');

if nargin < 1
    % error: not enough input arguments
    cd '..'
    error(['Usage: main( features_type, test_only, c, g )' 10 ...
        '- features_type: type of features to use' 10 ...
        '- test_only    : whether to make only the' 10 ...
        '                 test step, useful' 10 ...
        '                 if the features have' 10 ...
        '                 already been extracted' 10 ...
        '- c            : the c parameters for the SVM' 10 ...
        '- gamma        : the gamma parameters for the SVM']);
else
    % Different types of feature extraction
    if strcmp(features_type, 'rgb')
        
        if not(test_only)
            % extract facial images for each file
            extractor(false)
            
            % extract landmarks, normalize faces and extract LBP features
            landmarks(false)
            
            if isempty(c) || isempty(g)
                % compute best c and gamma for svm
                [c,g] = parameters(false);
            end
            
        elseif isempty(c) || isempty(g)
            warning(['no SVM parameters provided', 10, 'Using c=64 gamma=1']);
            c = 64;
            g = 1;
        end
        
        % test accuracy and plot ROC
        test(c,g);
        
    elseif strcmp(features_type, 'depth')
        
        if not(test_only)
            % extract facial images for each file
            extractor(true)
            
            % extract landmarks, normalize faces and extract LBP features
            landmarks(true)
            
            if isempty(c) || isempty(g)
                % compute best c and gamma for svm
                [c,g] = parameters(true);
            end
            
        elseif isempty(c) || isempty(g)
            warning(['no SVM parameters provided', 10, 'Using c=256 gamma=256']);
            c = 256;
            g = 256;
        end
        
        % test accuracy and plot ROC
        d_test(c,g);
        
    elseif strcmp(features_type, 'lbp-top')
        
        if not(test_only)
            % extract facial images for each file, landmarks,
            % normalize faces and extract LBP features
            extractor_LBPTOP
            
            if isempty(c) || isempty(g)
                % compute best c and gamma for svm
                [c,g] = parameters_LBPTOP;
            end
            
        elseif isempty(c) || isempty(g)
            warning(['no SVM parameters provided', 10, 'Using c=4 gamma=1']);
            c = 4;
            g = 1;
        end
        
        % test accuracy and plot ROC
        test_LBPTOP(c,g);
        
    else
        disp('invalid feature extraction type');
        disp('    -> rgb');
        disp('    -> depth');
        disp('    -> lbp-top');
    end
end
cd '..'
end

