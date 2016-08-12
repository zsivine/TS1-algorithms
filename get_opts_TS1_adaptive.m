function opts = get_opts_TS1_adaptive(maxr,m,n,sr,fr)

% Shuai Zhang
% April 2015


if sr <= 0.5*fr || fr >= 0.4
    hard = 1;
%     fprintf('This is a "hard" problem! \n\n');
else
    hard = 0;
%     fprintf('This is an "easy" problem! \n\n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if hard && max(m,n) < 1100
% hard
    opts.lambda = 1e-4;  % final lambda
    opts.tol = 1e-7;    % tolerance for subproblems in continuation
    opts.miu = 1;        % miu_0
    opts.mxitr = 1.5e+3;   % maximum iteration number for the outer loop

else
% easy    
    opts.lambda = 1e-4;
    opts.tol = 1e-7;
    opts.miu = 1;
    opts.mxitr = 1.5e+3;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
% opts.xs = xs;  % true solution xs is given
opts.fastsvd_ratio_leading = 5e-2; % ratio for computing the hard thresholding, i.e., the rank for next iteration

opts.eta = 1/4; % ratio to decrease lambda in continuation
opts.maxr = maxr; % maximum rank r is given 
opts.sr = sr;


