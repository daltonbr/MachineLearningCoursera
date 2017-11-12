function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
% Hint2: we might use element-wise multiplication operation (.*) and the
% sum operation (sum) when writing this function

h_theta = sigmoid(X * theta);

% Left and Right Hand Sides
LHS = -y' * log(h_theta);       % log() is a element-wise function
RHS = (1 - y)' * log(1 - h_theta);

% regularized J
J = (1/m) * (LHS - RHS);

% regularizing (we can safely modify theta, since this a local variable
% and we are not using it anymore)
theta(1) = 0;

theta_squared = theta' * theta;
% lambda = 1;

% regularizing term
regularized = (lambda/(2*m)) * theta_squared;

J = J + regularized;

% gradients
grad = (1/m) * (X' * (h_theta - y));

% theta(1) is already zero (from above operations)
% regularizated gradient term
reg_gradient = (lambda/m) * theta;

grad = grad + reg_gradient;

% =============================================================

grad = grad(:); % grad is returned as a column vector

end
