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

K = zeros(length(y),length(y));
for k_1 = 1:length(y)
    for k_2 = 1:length(y)
        K(k_1, k_2) = y(k_1) * y(k_2) * x(:, k_1).' * x(:, k_2);
    end
end

init_w_1 = 0.5;
w = [init_w_1; 1];
alpha = ones(1, length(y)) * 0.5;

lambda = 1;
iterate = 50;
etha = 0.001;

%% SVM
store_alpha = [];
store_w_hat = [];
store_sum_hinge_loss = [];
store_score_dual_lagrage = [];
for count = 1 : iterate
    alpha = projected_gradient(K, alpha, lambda, etha);
    w = w_hat_func(x, y, alpha, lambda);
    sum_hinge_loss = sum_hinge_loss_func(x, y, w, lambda);
    score_dual_lagrage = score_dual_lagrage_func(K, alpha, lambda);
    
    store_alpha = [store_alpha; alpha];
    store_w_hat = [store_w_hat, w];
    store_sum_hinge_loss = [store_sum_hinge_loss; sum_hinge_loss];
    store_score_dual_lagrage = [store_score_dual_lagrage; score_dual_lagrage];    
end

figure;
plot(1:iterate, store_sum_hinge_loss); hold on;
plot(1:iterate, store_score_dual_lagrage);
xlabel('Iteration'); ylabel('loss');
legend('hinge', 'dual lagrange','Location','northeast');

figure;
plot(1:iterate, store_sum_hinge_loss - store_score_dual_lagrage);
xlabel('Iteration'); ylabel('hinge - dual lagrange');

figure;
for l = 1:iterate
    plot(1:iterate, store_alpha(:, l)); hold on;
end
plot_data(x_1, x_2, w.');


%% function

%% (1) Sum of hinge loss function
function sum_hinge_loss = sum_hinge_loss_func(x, y, w, lambda)

sum_hinge_loss = 0;
for i = 1:length(y)
    hinge = [0, 1 - y(i) * w.' * x(:, i)];
    sum_hinge_loss = sum_hinge_loss + max(hinge);    
end
sum_hinge_loss = sum_hinge_loss + lambda * w.' * w;

end

%% (2) Score of the dual Lagrange function
function score_dual_lagrage = score_dual_lagrage_func(K, alpha, lambda)

score_dual_lagrage = -1 / (4 * lambda) * alpha * K * alpha.' + alpha * ones(length(alpha), 1);

end

%% (3) w^hat
function w_hat = w_hat_func(x, y, alpha, lambda)

w_hat = zeros(2, 1);
for i = 1:length(y)
    w_hat = w_hat + alpha(i) * y(i) * x(:, i);
end
w_hat = 1/ (2 * lambda) * w_hat;

end

%% Projected gradient
function alpha_new = projected_gradient(K, alpha, lambda, etha)

alpha = alpha.';
p = alpha - etha * (1 / (2 * lambda) * K * alpha - ones(length(alpha), 1));

for i = 1:length(p)
    if p(i) > 1
        p(i) = 1;
    elseif p(i) < 0
        p(i) = 0;
    end
end

alpha_new = p.';

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