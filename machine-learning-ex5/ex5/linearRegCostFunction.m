function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% linear equation and hence theta 3 and above are zero
% theta0 + theta1 * X;
% theta(3:end) = 0
hypothesis = X * theta;
J = 1/(2*m) .* sum( (hypothesis - y)' * (hypothesis - y) );
% with regularization term
regularizationTerm = (lambda * sum( theta(2:end)' * theta(2:end))) / (2*m);
J = J + regularizationTerm;

% difference between hypothesis and actual result
diff = (hypothesis - y);
% =============================================================
grad = 1 /m * ( X' * diff);
grad(2:end) = grad(2:end) + (lambda*theta(2:end)) / m;










% =========================================================================

grad = grad(:);

end
