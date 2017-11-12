function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       
%      X         is (m x n)          - e.g. (5000 x 400)
%      all_theta is (num_labels x n) - e.g. (10 x 400)

% is shape (m x num_labels) - e.g. (5000x10)
all_classifiers = sigmoid (X * all_theta');

% c1 = size(all_classifiers, 1);
% c2 = size(all_classifiers, 2);
% fprintf('all_classifiers size:[%dx%d]\n', c1, c2);

%    If called with one input and two output arguments, `max' also
%      returns the first index of the maximum value(s).  Thus,
%
%           [x, ix] = max ([1, 3, 5, 2, 5])
%               =>  x = 5
%                   ix = 3

% max will the get the maximum value at each column - and it's index
% i.e. the best probability for each label
% max_classifier_row is (m x 1) - e.g. (5000 x 1)
[max_classifier_row, max_index] = max(all_classifiers, [], 2);

% m1 = size(max_classifier_row, 1);
% m2 = size(max_classifier_row, 2);
% fprintf('max_classifier_row size:[%dx%d]\n', m1, m2);

% fprintf('max_index:%d\n', max_index);
% m1 = size(max_index, 1);
% m2 = size(max_index, 2);
% fprintf('max_index size:[%dx%d]\n', m1, m2);

p = max_index;

% =========================================================================


end
