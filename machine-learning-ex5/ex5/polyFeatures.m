function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

% ... where vector1 is a column vector of the feature values 'X', 
% and vector2 is a row vector of exponents from 1 to 'p'.


vector1 = X(:, 1);
vector2 = [];
for i = 1:p
    vector2 = [vector2 i];
end
X_poly = bsxfun(@power, vector1, vector2);




% =========================================================================

end
