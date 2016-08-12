% This file gives examples how to use TS1 to solve matrix completion
% problems.
% 
% Author: Shuai Zhang 
% Time: Mon 23 Feb 2015 07:12:51 PM PST 
%--------------------------------------------------------------------------
clear; clc;
close all;

% prepare data
% X is m by n matrix, sr is the sampling ratio, p is the number of samples,
% r is the rank of the matrix M to be completed.

m = 400; n = 400; sr = 0.5; p = round(m*n*sr); r = 16;   
% m = 100; n = 100; sr = 0.5; p = round(m*n*sr); r = 10;   
% m = 100; n = m ; p = 5666; sr = p/m/n; r = 10;
% m = 200; n = m; p = 15665; sr = p/m/n; r = 15;
% m = 500; n = m; p = 49471; sr = p/m/n; r = 10;
% m = 150; n = 300; sr = 0.49; p = round(m*n*sr); r = 10;

% fr is the freedom of set of rank-r matrix, maxr is the maximum rank one
% can recover with p samples, which is the max rank to keep fr < 1
fr = r*(m+n-r)/p; maxr = floor(((m+n)-sqrt((m+n)^2-4*p))/2);

% get problem
A = randperm(m*n); A = A(1:p); % A gives the position of samplings

% v1: iid Gaussian
% xl = randn(m,r); xr = randn(n,r); xs = xl*xr'; % xs is the matrix to be completed
% v2: coherent Gaussian 
cohe = 0.3; % range (0,1) 
sigma = cohe*ones(r); sigma = sigma + (1-cohe)*eye(r); 
mu = ones(1,r); 
xl = mvnrnd(mu,sigma,m);  xr = mvnrnd(mu,sigma,n); xs = xl*xr';
% v3: uniform distribution
% xl = rand(m,r)-0.5; xr = rand(n,r)-0.5; 
% xs = xl*xr';


b = reshape(xs,m*n,1); b = b(A); % b is the samples from xs

% get parameters for TS1
option = get_opts_TS1(maxr,m,n,sr,fr); 

% call TS1 to solve the matrix completion problem
fprintf('solving by Threshold TS1....\n');
% option.est_rank = 1;  K = round(1.5*r); option.a = 100;

option.est_rank = 0; K = r; option.a = 1;



solve_fpc = cputime;
Out = TS1_s1_v2(m,n,A,b',K,option);
%Out = TS1_adaptive_Zhang(m,n,A,b',K,option);
solve_fpc = cputime - solve_fpc; Out.solve_t = solve_fpc;

fprintf('done!\n');

% print the statistics
fprintf('m = %d, n = %d, r = %d, p = %d, \nsamp.ratio = %3.2f, freedom = %3.2f \n',...
         m,n,r,p,sr,fr);
fprintf('Time = %3.2f, relative error = %3.2e\n', ...
         solve_fpc, norm(Out.x-xs,'fro')/norm(xs,'fro'));
fprintf('Rank of original matrix: %d, \n and rank of numerical result: %d \n',rank(xs),Out.rank)
fprintf('Iteration times: %d \n',Out.iter)


