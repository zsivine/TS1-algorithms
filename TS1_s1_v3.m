%--------------------------------------------------------------------------
% Threshold TS1 Method based on Approximate SVD for Matrix Completion Problem. 
%
% Solves
%           min  rank(X)
%           s.t. X_ij = M_ij , (i,j) \in \Omega
% 
% Author: Shuai Zhang 
% Date: Feb 2015
% version: random svd --- fazel

%--------------------------------------------------------------------------
% opts: 
%      a:  parameter for TS1(TL1)
%      mu(lambda): regularization parameter
%      tau(miu): parameter for operator B_miu(.) 
%      est_rank: 0 known r; 1 decreasing rank estimate; 2 increasing rank 
% r: rank estimation for truth solution
%
% Out: 
%      x: numerical result
%      iter: iteration number
%      rank: rank for numerical matrix
%      obj: residual for each iteration

function Out = TS1_s1_v3(m,n,A,b,r,opts)

rk = r;

Atb = AtMVMx(A,b,m,n);  % calculate AtMb

% initial value
% opts.x0 = zeros(m,n);
opts.x0 = Atb;

lambda = opts.lambda; 
a = opts.a;


if max(m,n) > 1000
    
%     [~,sigma] = LinearTimeSVD(Atb,min([m/2,n/2,1000]),1,ones(n,1)/n); %v1
    [~,sigma,~] = svds(Atb,1); % v2
        
    nrm2Atb = sigma(1);
else
    nrm2Atb = norm(Atb);
end


lambdaf = lambda; % final value of mu
x = opts.x0; miu = opts.miu;
% intial value for lambda
% lambda = nrm2Atb*opts.eta; 
lambda = nrm2Atb*min(3,1/opts.sr); 

pp = ones(n,1)/n; 
sn = min([m/2,n/2,round(2*opts.maxr-2)]);

g = get_g(x,m,n,A,Atb); % A^t(Ax - b)



%% main loop
i = 0; err = 1;
%-------------------------------------------
itr_rank = 0; minitr_reduce_rank = 5;    maxitr_reduce_rank = 50;  
est_rank = 1;
Objv = zeros(opts.mxitr,1);
if isfield(opts,'est_rank');    est_rank= opts.est_rank;   end
datanrm = max(1,norm(b));  

while (i <= opts.mxitr) && (err >= opts.xtol)
    itr_rank = itr_rank + 1;
    i = i + 1;
    
    xp = x;
    y = x - miu*g; % y = B_miu(x)
    
    if i == 1
        if max(m,n) < 1000
            [U,S,V] = svd(y); sigma = diag(S);
        else
            
%             [U,sigma] = LinearTimeSVD(y,sn,opts.maxr,pp); % v1
%             [U,S,~] = svds(y, opts.maxr); sigma = diag(S); % v2
            [U,S,~] = rand_svd(y,opts.maxr,1,1,1); sigma = diag(S); % v3
            
            invsigma = zeros(size(sigma)); indx = find(sigma);
            invsigma(indx) = 1./sigma(indx);
            V = (diag(invsigma)*U'*y)';
        end
    else
        
        sp = s(s>0); mx = max(sp);
        
        % number of singular values to approximate
        kk = length(find(sp > mx*opts.fastsvd_ratio_leading));
        kk = max(r+1,min(kk,sn)); % kk >= r+1
        
%         [U,sigma] = LinearTimeSVD(y,sn,kk,pp);
%         [U,sigma] = LinearTimeSVD(y,max(sn,kk+2),kk,pp); % v1
%         [U,S,~] = svds(y, kk); sigma = diag(S); % v2
        [U,S,~] = rand_svd(y,kk,1,1,1); sigma = diag(S); % v3
        
        
        invsigma = zeros(size(sigma)); indx = find(sigma);
        invsigma(indx) = 1./sigma(indx);
        V = (diag(invsigma)*U'*y)';
    end
    
    s = sigma; % sigma is a vector
    ind = find(s > 0);
    Ue = U(:,ind); Ve = V(:,ind); s = s(ind); 
%----------------------------------------------
    if est_rank >= 1; rank_estimator_adaptive(); end
    if rk ~= r; r = rk; end
    r = min(r,length(s)-1); % v3    
    
    
    %% TL1 Thresholding ... version 1
    lambda2 = (a+2*s(r))^2/8/(a+1);
    lambda1 = a*s(r+1)/(a+1);
    
    if lambda1 <= a^2/2/(a+1)
%        fprintf('threshold scheme 1...... \n');
       lambda = max(lambdaf, min(lambda1/miu, lambda));
%        lambda = lambda2/miu;
       t2 = lambda*miu*(a+1)/a;
       idx = find ( s >= t2);
       x = zeros(length(s),1); phi = x; 
       phi(idx) = acos( max( 1 - 27/2*lambda*miu*(a+1)*a ./ (a + s(idx)).^3,-0.99));
       x(idx)=  2*(a+s(idx)).*cos(phi(idx)/3)/3 - 2*a/3 + s(idx)/3 ;
    else
%        fprintf('threshold scheme 2...... \n');
       lambda = max(lambdaf, min(lambda2/miu,lambda));
%        lambda = lambda1/miu;
       t1 = sqrt(2*lambda*miu*(a+1)) - a/2;
       idx = find ( s > t1);
       x = zeros(length(s),1); phi = x; 
       phi(idx) = acos( max( 1 - 27/2*lambda*miu*(a+1)*a ./ (a + s(idx)).^3,-0.99));
       x(idx)= 2*(a+s(idx)).*cos(phi(idx)/3)/3 - 2*a/3 + s(idx)/3 ;
    end
    
    S = diag(x); % S is a diagonal matrix
    x = Ue*S*Ve';
    s = diag(S);
    g = get_g(x,m,n,A,Atb);
    
    % check shrinkage
    nrmxxp = norm(x - xp,'fro');
    err = nrmxxp/max(norm(xp,'fro'),1);
    
    resid = norm(b-x(A)); % residual
    Objv(i) = resid/datanrm;
    
%     if lambda == lambdaf
%        opts.mxitr = 1500; 
%     end

end

% x(A) = b;
Out.x = x; 
Out.iter = i;
Out.rank = rank(x);
Out.obj = Objv(1:i);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTION FOR CALCULATING g
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


% SUBFUNCTION FOR RANK ESTIMATE
    function rank_estimator_adaptive()
            rk_jump = 10; 
            rank_min = 5;
            % dR = abs(diag(R));  % QR factorization
            % drops = dR(1:end-1)./dR(2:end); 
            drops = s(1:end-1)./s(2:end); 
            
            [dmx,imx] = max(drops);  rel_drp = (r-rank_min-1)*dmx/(sum(drops)-dmx);
            %imx
            %bar(drops)
            if (rel_drp > rk_jump && itr_rank > minitr_reduce_rank) ...
                   || itr_rank > maxitr_reduce_rank; %bar(drops); pause;
                
                rk = max([imx, floor(0.1*r), rank_min]);                
                est_rank = 0; itr_rank = 0; 

            end
    end %rank

end % main