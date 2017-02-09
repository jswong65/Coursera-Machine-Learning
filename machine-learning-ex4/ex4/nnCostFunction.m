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


%part 1:

% Calculate prediction using Feedforward algorithm (from week 4 predict.m)
tmp_X = [ones(m,1) X];

% calculate units of second layer
unit1 = sigmoid(tmp_X * transpose(Theta1));

% add bias term (1)
unit1 = [ones(m,1) unit1];

% calculate units of output layer
pred = sigmoid(unit1 * transpose(Theta2));

% convert label y to [1,0,0, ...] array format
tmp_y = zeros(length(y), num_labels);
for i = 1:length(y)
	tmp_y(i,y(i)) = 1;
end 

% calculate cost function
for i = 1:m
	for k = 1:num_labels 
		J += (-tmp_y(i,k)*log(pred(i,k)) - (1-tmp_y(i,k))*log(1-pred(i,k))); 
	end
end

J /= m;


% calculate regularization terms, should skip theta terms for bias
reg_terms = 0.;
reg_terms = sum(sum(Theta1(:, 2:end) .* Theta1(:, 2:end))) + ...
sum(sum(Theta2(:, 2:end) .* Theta2(:, 2:end)));
reg_terms = reg_terms * lambda / (2*m);
J = J + reg_terms;

% Part 2 implement backpropagation
%fprintf('dimension of Theta1 %d,%d\n', size(Theta1));
%fprintf('dimension of Theta2 %d,%d\n', size(Theta2));
%fprintf('dimension of Theta1_grad %d,%d\n', size(Theta1_grad));
%fprintf('dimension of Theta2_grad %d,%d\n', size(Theta2_grad));

for i =1:m
	% Feedforward for each X
	a_1 = tmp_X(i,:);
	%fprintf('dimension of a_1 %d,%d\n', size(a_1));
	z_2 = a_1*transpose(Theta1);
	%fprintf('dimension of z_2 %d,%d\n', size(z_2));
	a_2 = [1 sigmoid(z_2)];
	%fprintf('dimension of a_2 %d,%d\n', size(a_2));
	z_3 = a_2*transpose(Theta2);
	%fprintf('dimension of z_3 %d,%d\n', size(z_3));
	a_3 = sigmoid(z_3);
	d_3 = a_3(:) - tmp_y(i,:)(:);
	%fprintf('dimension of d_3 %d,%d\n', size(d_3));
	d_2 = (transpose(Theta2)*d_3)(2:end) .* sigmoidGradient([z_2(:)]);
	%fprintf('dimension of d_2 %d,%d\n', size(d_2));
	
	delta_2 = d_3*a_2;
	%fprintf('dimension of delta_2 %d,%d\n', size(delta_2));
	delta_1 = d_2*a_1;
	%fprintf('dimension of delta_1 %d,%d\n', size(delta_1));
	Theta2_grad = Theta2_grad + delta_2;
	Theta1_grad = Theta1_grad + delta_1;
end

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;


% add regularization terms for gradients
tmp_Theta1 = Theta1;
tmp_Theta2 = Theta2;
tmp_Theta1(:,1) = 0 
tmp_Theta2(:,1) = 0 

Theta1_grad += (lambda/m)*tmp_Theta1;
Theta2_grad += (lambda/m)*tmp_Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
