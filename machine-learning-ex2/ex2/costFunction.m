function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

h_theta = sigmoid(X * theta);

% Left and Right Hand Sides
LHS = -y' * log(h_theta);
RHS = (1 - y)' * log(1 - h_theta);

% regularized J
J = (1/m) * (LHS - RHS);

% regularizing (we can safely modify theta, since this a local variable
% and we are not using it anymore)
theta(1) = 0;

theta_squared = theta' * theta;
lambda = 1;

regularized = (lambda/2*m) * theta_squared;

%J = J + regularized;

% gradients
grad = (1/m) * (X' * (h_theta - y));

% theta(1) is already zero (from above operations)
% regularizated gradient term
%reg_gradient = (lambda/m) * theta;

%grad = grad + reg_gradient;




% =============================================================

end
