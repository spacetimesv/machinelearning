function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%function [model] = svmTrain(X, Y, C, kernelFunction, ...
%                            tol, max_passes)
%SVMTRAIN Trains an SVM classifier using a simplified version of the SMO 
%algorithm. 
%   [model] = SVMTRAIN(X, Y, C, kernelFunction, tol, max_passes) trains an
%   SVM classifier and returns trained model. X is the matrix of training 
%   examples.  Each row is a training example, and the jth column holds the 
%   jth feature.  Y is a column matrix containing 1 for positive examples 
%   and 0 for negative examples.  C is the standard SVM regularization 
%   parameter.  tol is a tolerance value used for determining equality of 
%   floating point numbers. max_passes controls the number of iterations
%   over the dataset (without changes to alpha) before the algorithm quits.

% values for C and sigma
% (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30).
minError=0;
firstIteration = 1;
for c = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    for tempSigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

        if tempSigma == c
           continue; 
        end
        
        % for each pair of C and sigma from above set
        % choose a pair of values from the set
        % get the svn trained model
        model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, tempSigma)); 
        predictions = svmPredict(model, Xval);
        error = mean( double(predictions ~= yval) );
        
        % assign the first error
        if firstIteration == 1
           minError = error; 
           firstIteration = 0;
        end
        
        % save minimum error in minError variable
        if error < minError
           minError = error; 
           C = c;
           sigma = tempSigma;
        end
    end
    
end




% =========================================================================

end
