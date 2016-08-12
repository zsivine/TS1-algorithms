%--------------------------------------------------------------------------
% Threshold TS1 Method based on Approximate SVD for Matrix Completion Problem. 
%
% Solves
%           min  rank(X)
%           s.t. X_ij = M_ij , (i,j) \in \Omega
% 
% Author: Shuai Zhang 
% Date: Feb 2015
%--------------------------------------------------------------------------

% opts: 
%      a:  parameter for TS1(TL1)
%      mu(lambda): regularization parameter
%      tau(miu): parameter for operator B_miu(.) 
% r: rank estimation for truth solution

function Out = TS1_s1(m,n,A,b,r,opts)

Atb = AtMVMx(A,b,m,n);  % calculate AtMb

% initial value
opts.x0 = zeros(m,n);
% opts.x0 = Atb;

a = opts.a;

if max(m,n) > 1000
    [~,sigma] = LinearTimeSVD(Atb,min([m/2,n/2,1000]),1,ones(n,1)/n);
    nrm2Atb = sigma(1);
else
    nrm2Atb = norm(Atb);
end


x = opts.x0; 
miu = opts.miu;
lambdaf = opts.lambda; % final value of mu
% intial value for lambda
% lambda = nrm2Atb*opts.eta; 
lambda = nrm2Atb*min(3,1/opts.sr); 
if lambda < lambdaf, 
    lambda = lambdaf;
    opts.maxinneriter = 500;
end


innercount = 0;
pp = ones(n,1)/n; sn = min([m/2,n/2,round(2*opts.maxr-2)]);
g = get_g(x,m,n,A,Atb); % A^t(Ax - b)

%% main loop
for i = 1:opts.mxitr
    
    xp = x;
    y = x - miu*g; % y = B_miu(x)
    
    if i == 1
        if max(m,n) < 1000
            [U,S,V] = svd(y); sigma = diag(S);
        else
            [U,sigma] = LinearTimeSVD(y,sn,opts.maxr,pp);
            invsigma = zeros(size(sigma)); indx = find(sigma);
            invsigma(indx) = 1./sigma(indx);
            V = (diag(invsigma)*U'*y)';
        end
    else
        
        sp = s(s>0); mx = max(sp);
        
        % number of singular values to approximate
        kk = length(find(sp > mx*opts.fastsvd_ratio_leading));
        kk = max(r+1,min(kk,sn)); % kk >= r+1
        
        [U,sigma] = LinearTimeSVD(y,sn,kk,pp);
        invsigma = zeros(size(sigma)); indx = find(sigma);
        invsigma(indx) = 1./sigma(indx);
        V = (diag(invsigma)*U'*y)';
    end
    
    s = sigma; % sigma is a vector
    ind = find(s > 0);
    Ue = U(:,ind); Ve = V(:,ind); s = s(ind); 
%     S = diag(s); % here S is a diagonal matrix 
    
    
    %% TL1 Thresholding ... version 1
    lambda2 = (a+2*s(r))^2/8/(a+1);
    lambda1 = a*s(r+1)/(a+1);
    
    if lambda1 <= a^2/2/(a+1)
%        fprintf('threshold scheme 1...... \n');
       lambda = max(lambdaf, min(lambda1/miu, lambda));
%        lambda = lambda2/miu;
       t2 = lambda*miu*(a+1)/a;
       idx = find ( s >= t2);
    else
%        fprintf('threshold scheme 2...... \n');
       lambda = max(lambdaf, min(lambda2/miu,lambda));
%        lambda = lambda1/miu;
       t1 = sqrt(2*lambda*miu*(a+1)) - a/2;
       idx = find ( s > t1);
    end
    
    x = zeros(length(s),1); phi = x; 
    phi(idx) = max( 1 - 27/2*lambda*miu*(a+1)*a ./ (a + s(idx)).^3 , ...
                    -1) ;
    phi(idx) = acos(phi(idx));
    x(idx)=  2*(a+s(idx)).*cos(phi(idx)/3)/3 - 2*a/3 + s(idx)/3 ;
    
    

%%     % TL1 Thresholding ... version 2
%     lambda2 = (a+2*s(r))^2/8/(a+1);
%     lambda1 = a*s(r+1)/(a+1);
%     lambda2 = max(lambdaf, min(lambda2/miu, lambda));
%     lambda1 = max(lambdaf, min(lambda1/miu, lambda));
%     
%     if lambda1 <= a^2/2/(a+1)/miu
%        lambda = lambda1;
% %        fprintf('threshold scheme 1...... \n');
%        t2 = lambda*miu*(a+1)/a;
%        idx = find ( s >= t2);
%        x = zeros(length(s),1); phi = x; 
%        phi(idx) = acos( max( 1 - 27/2*lambda*miu*(a+1)*a ./ (a + s(idx)).^3,-0.99));
%        x(idx)=  2*(a+s(idx)).*cos(phi(idx)/3)/3 - 2*a/3 + s(idx)/3 ;
%     else
% %        fprintf('threshold scheme 2...... \n');
%        lambda = lambda2;
%        t1 = sqrt(2*lambda*miu*(a+1)) - a/2;
%        idx = find ( s > t1);
%        x = zeros(length(s),1); phi = x; 
%        phi(idx) = acos( max( 1 - 27/2*lambda*miu*(a+1)*a ./ (a + s(idx)).^3,-0.99));
%        x(idx)= 2*(a+s(idx)).*cos(phi(idx)/3)/3 - 2*a/3 + s(idx)/3 ;
%     end

    S = diag(x);
    x = Ue*S*Ve';
    s = diag(S);
    g = get_g(x,m,n,A,Atb);
    
    % check shrinkage
    nrmxxp = norm(x - xp,'fro');

    critx = nrmxxp/max(norm(xp,'fro'),1); 
    innercount = innercount + 1;
    
    if lambda == lambdaf
        opts.maxinneriter = 500;
    end

    if (critx < opts.xtol) || (innercount >= opts.maxinneriter)
        innercount = 0;
        % stop if reached lambdaf
        if lambda == lambdaf
            Out.x = x; Out.iter = i;
            return
        end
        
        lambda = opts.eta*lambda;
        lambda = max(lambda,lambdaf);
        
        if lambda == lambdaf
            opts.maxinneriter = 500;
        end
    end

end


% did not converge within opts.mxitr
Out.x = x; Out.iter = i;

%--------------------------------------------------------------------------
% SUBFUNCTION FOR CALCULATING g
%--------------------------------------------------------------------------
    function y = AMVMx(A,x)
        % y = A*x
        y = x(A);
    end

    function y = AtMVMx(A,b,m,n)
        % y = A'*b
        y = zeros(m*n,1);
        y(A) = b;
        y = reshape(y,m,n);
    end


    function g = get_g(x,m,n,A,Atb)
        Ax = AMVMx(A,reshape(x,m*n,1));
        g = AtMVMx(A,Ax,m,n)-Atb;
    end
end
