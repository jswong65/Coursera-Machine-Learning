function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

for i=1:m
	z = dot(X(i,:), theta);
	J += (-y(i)*log(sigmoid(z)) - (1-y(i))*log(1-sigmoid(z))); 
end
J = (J + (lambda/2) * sum(theta(2:length(theta),:).^2) ) / m
z = X * theta;
error = sigmoid(z)-y;
grad = (1/m) * transpose(X) * error;
tmp_theta = theta;
tmp_theta(1) = 0;
grad += (lambda/m) * tmp_theta;


% =============================================================

end
