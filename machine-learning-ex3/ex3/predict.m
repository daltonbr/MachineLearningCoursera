function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add a column of ones to the X data matrix (bias unit)
a1 = [ones(m, 1) X]; 

% Theta1 (25 x 401) (h x n) - where 'h' is the number of hidden units
% a1     (5000 x 401) (m x n) - where 'n' is the number of features including the bias unit
% z2     (m x h)
z2 = a1 * Theta1'; % (m x n) * (n * h) = (m * h)
z2_sig = sigmoid(z2);

% Add a column of ones to the a1 (bias unit)
a2 = [ones(m, 1) z2_sig];  % (m x (h+1))

% Theta2 (10 x 26) (c x (h+1)) - where 'c' is the number of labels
% a2     (m x (h+1))
z3 = a2 * Theta2'; %?
a3 = sigmoid(z3); % a3 is (m x c)

[max_classifier_row, max_index] = max(a3, [], 2);
% p is a vector of size (m x 1)
p = max_index;

% =========================================================================

end
