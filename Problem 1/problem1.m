%% Initialize
clear; close all;

%% Generate the Data
num_data = 50;
mu_1 = [0 2];
sigma_1 = [1 0; 0 3];
R_1 = chol(sigma_1);
x_1 = (repmat(mu_1,num_data,1) + randn(num_data,2)*R_1).';

mu_2 = [0 -2];
sigma_2 = [1 0; 0 3];
R_2 = chol(sigma_2);
x_2 = (repmat(mu_2,num_data,1) + randn(num_data,2)*R_2).';

%% Initialize
x = [x_1 x_2]; 
y = [ones(1,num_data), zeros(1,num_data)];
init_w_1 = 0.5;
w = [init_w_1; 1];

alpha = 0.001;
lambda = 0.01;
iterate = 100;

%% Batch Steepest Gradient Method
[w_bsg, J_bsg] = batch_steepest_gradient(x, y, w, alpha, lambda, iterate);
% Plot the line
plot_data(x_1, x_2, w_bsg);

%% Newton Method
[w_newton, J_newton] = newton(x, y, w, alpha, lambda, iterate);
% Plot the line
plot_data(x_1, x_2, w_newton);

%% Compare
figure;
plot(J_bsg(:, 2)); hold on;
plot(J_newton(:, 2));
xlabel('Iteration'); ylabel('Regularized loss');
legend('Batch Steepest Gradient', 'Newton','Location','southeast');


%% function

%% Sigmoid function
function P = sigmoid(x, y, w)

P = 1 / (1 + exp(-y * (w.' * x).'));

end

%% Cost function
function J = cost(x, y, w, lambda)

J = zeros(2,1);
size_y = size(y);
for i = 1:size_y(2)
    J = J + log(1 + exp(-y(i) * (w.' * x(:, i)).'));
end
J = -1/size_y(2) .* J + 2 * lambda * w;
    
end

%% Gradient function
function dJ = gradient(x, y, w, lambda)

dJ = zeros(2,1);
size_y = size(y);
for i = 1:size_y(2)
    dJ = dJ + (1 - sigmoid(x(:,i), y(i), w) ) * y(i) * x(:,i);
end
dJ = -1/size_y(2) .* dJ + 2 * lambda * w;
    
end

%% Hessian function
function H = hessian(x, y, w, lambda)

H = zeros(2,2);
size_y = size(y);
for i = 1:size_y(2)
    H = H + sigmoid(x(:,i), y(i), w) * (1 - sigmoid(x(:,i), y(i), w)) * y(i).^2 * x(:,i) * x(:,i).';
end
H = -1/size_y(2) .* H + 2 * lambda;

end

%% Batch Steepest Gradient Method function
function [w_bsg, J_bsg] = batch_steepest_gradient(x, y, w, alpha, lambda, iterate)

w_bsg = [];
J_bsg = [];

for count = 1 : iterate
    % Gradient
    dJ = gradient(x, y, w, lambda);
    % Update w
    w = w + alpha * dJ;     
    % Store the data
    w_bsg = [w_bsg; w(1), w(2)];
    J = cost(x, y, w, lambda);
    J_bsg = [J_bsg; J(1), J(2)];      
end

end

%% Newton Method function
function [w_newton, J_newton] = newton(x, y, w, alpha, lambda, iterate)

w_newton = [];
J_newton = [];

for count = 1 : iterate
    % Gradient and Hessian
    dJ = gradient(x, y, w, lambda);
    H = hessian(x, y, w, lambda);
    % Update w
    d = -pinv(H) * dJ;
    w = w + alpha * d;    
    % Store the data
    w_newton = [w_newton; w(1), w(2)];
    J = cost(x, y, w, lambda);
    J_newton = [J_newton; J(1), J(2)];  
end

end

%% Plot the data fuction
function plot_data(x_1, x_2, w)
figure;
plot(x_1(1,:), x_1(2,:), 'ro'); hold on;
plot(x_2(1,:), x_2(2,:), 'bx'); hold on;
xlabel('{x_1}'); ylabel('{x_2}');

x1_w = -3:0.5:3;
x2_w = - w(end, 1)/w(end, 2) * x1_w;
plot(x1_w, x2_w, 'g');
end