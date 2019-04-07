function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);
m = size(X, 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% distance matrix for storing squared min distance between each training 
% example with each centroids
distance = zeros(m, K);

for i = 1:m
    distance = zeros( size(centroids, 1), 1);
    %for k = 1:K
        %for indexX = 1:m

            % diff from x to centroid point
            % D = (centroids(index, 1) - X(indexX, :));
            % distance = bsxfun(@minus, centroids(index,:), X);

        %end
        %distance(k)  = 

    %end
    %[minSum, idx(:, 1) ] = min(sum(distance.^2, 2));
    
    for k = 1:K
		distance(k, :) = sum((X(i, :) - centroids(k, :)) .^ 2, 2 );
	end
	[x, index] = min(distance);
	idx(i) = index;
end





% =============================================================

end

