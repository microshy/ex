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
n=input_layer_size;
h=hidden_layer_size;
k=num_labels;
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
Y=zeros(m,num_labels);%首先要将y化为向量的形式
%将y化为向量的具体步骤
for i=1:m
    for j=1:k
        if j==y(i)
            Y(i,j)=1;
        end
    end
end
%计算A1
A1=[ones(m,1) X];%然后依照公式，一层一层计算输出量 5000*401
%计算A2
Z2=A1*Theta1';%5000*25
A2=[ones(m,1) sigmoid(Z2)];%5000*26
%计算A3
Z3=A2*Theta2';%5000*10
A3=sigmoid(Z3);%5000*10
%计算代价
J = -1/m*(ones(1,m)*(log(A3).*Y+log(1-A3).*(1-Y))*ones(k,1))...
    +lambda/2/m*(ones(1,h)*(Theta1(:,2:n+1).*Theta1(:,2:n+1))*ones(n,1)...
    +ones(1,k)*(Theta2(:,2:h+1).*Theta2(:,2:h+1))*ones(h,1));
    %照抄公式，平方和不要计算首项theta

%在计算J之前已经利用前向网络计算了a(L)

delt3=A3-Y;%计算delt(L) 5000*10
%按照公式依次计算delt(L-1)到delt(2)
delt2=delt3*Theta2(:,2:h+1).*sigmoidGradient(Z2);%5000*25
%根据公式依次计算Det,最后一层不需要计算
Det2=delt3'*A2;%10*26
Det1=delt2'*A1;%25*401
%根据公式和原理，写出梯度的矩阵形式的计算公式
Theta2_grad(:,1)=Det2(:,1)./m;%10*1
Theta2_grad(:,2:h+1)=Det2(:,2:h+1)/m+lambda/m*Theta2(:,2:h+1);%10*25
Theta1_grad(:,1)=Det1(:,1)./m;%25*1s
Theta1_grad(:,2:n+1)=Det1(:,2:n+1)/m+lambda/m*Theta1(:,2:n+1);%25*400










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
