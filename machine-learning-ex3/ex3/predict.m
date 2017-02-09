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

disp(size(Theta1));
disp(size(Theta2));

% add bias term (1)
tmp_X = [ones(m,1) X];
disp(size(tmp_X));

% calculate units of second layer
unit1 = sigmoid(tmp_X * transpose(Theta1));
disp(size(unit1)); 

% add bias term (1)
unit1 = [ones(m,1) unit1];

% calculate units of output layer
output = sigmoid(unit1 * transpose(Theta2));
disp(size(output));

% select the highest prob in each row, the result of different one vs all classifiers. 
[max_val, max_ind] = max(output, [], 2);
p = max_ind;
disp(p);


% =========================================================================


end
