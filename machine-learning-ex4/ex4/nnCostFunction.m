function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% part 1 feed forward

% convert y to a binary valued position
% for example if size(y) = 10
% and for output 5 we represent as y = 0 0 0 0 1 0 0 0 0 0
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

% Determine activation values for each layer 
a1 = [ones(size(X, 1), 1) X];
z2 = a1 * Theta1';
hz2 = sigmoid(z2);

a2 = [ones(size(hz2, 1), 1) hz2];
a3 = sigmoid(a2 * Theta2');


% 3 - Compute the unregularized cost according to ex4.pdf (top of Page 5), 
% using a3, your y_matrix, and m (the number of training examples). 
% Note that the 'h' argument inside the log() function is exactly a3. 
% Cost should be a scalar value. Since y_matrix and a3 are both matrices, 
% you need to compute the double-sum.
% Remember to use element-wise multiplication with the log() function. 
% For a discussion of why you can't (easily) use matrix multiplication here, see this thread:
% for i = 1:hidden_layer_size
% size(y_matrix)
% size(a3)
innerSum = sum( (-y_matrix .* log(a3)) - ((1 - y_matrix) .* log(1 - a3)), 2 );
J =  1/m .* sum( innerSum );


% add the regularization terms
% Theta1(:, 2:end) Theta2(:, 2:end)

% with regularization term
% regularizationTerm = (lambda * sum( theta(2:end)' * theta(2:end))) / (2*m);
reg1 = sum( Theta1(:, 2:end) .^ 2 );
reg2 = sum( Theta2(:, 2:end) .^ 2 );
regularizationTerm = lambda * (sum(reg1) + sum(reg2)) / (2 * m);

J = J + regularizationTerm;

% back propagation
% set bias unit to activations
% difference between hypothesis and actual result
d3 = a3 - y_matrix;

% 4: ?2 or d2 is tricky. It uses the (:,2:end) columns of Theta2. 
% d2 is the product of d3 and Theta2(no bias), then element-wise scaled by sigmoid gradient of z2. 
% The size is (m x r) ? (r x h) --> (m x h). The size is the same as z2, as must be.
d2 = (d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);

% 5: ?1 or Delta1 is the product of d2 and a1. The size is (h x m) ? (m x n) --> (h x n)
size(d2)
size(a1)
Delta1 = d2' * a1;

% 6: ?2 or Delta2 is the product of d3 and a2. The size is (r x m) ? (m x [h+1]) --> (r x [h+1])
Delta2 = d3' * a2;

% 7: Theta1_grad and Theta2_grad are the same size as their respective Deltas, just scaled by 1/m.
Theta1_grad = Delta1 ./ m;
Theta2_grad = Delta2 ./ m;












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
