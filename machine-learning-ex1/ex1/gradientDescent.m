function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

minJ = 0;
minTheta = theta;
alpha = 0.01;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    prediction = X * theta;
    diff = (prediction - y);
    % tmpX = X;
    % tmpX(:, [1]) = [];

    gradient = (alpha/m) * X' * diff;
    theta = theta - gradient;

    % ============================================================

    % Save the cost J in every iteration    
    
    J_history(iter) = computeCost(X, y, theta);
    
    % set the first element as min
    if( iter == 1),
	minJ = J_history(iter);
    end
    if( minJ >= J_history(iter) ),
	minJ = J_history(iter);
	minTheta = theta;
	% iter
	% minJ
        % disp("Setting theta to ");
	% minTheta
    end

end

theta = minTheta;

end
