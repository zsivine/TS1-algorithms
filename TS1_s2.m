%--------------------------------------------------------------------------
% Full Adaptive Threshold TS1 Method based on Approximate SVD 
% for Matrix Completion Problem. 
%
% Solves
%           min  rank(X)
%           s.t. X_ij = M_ij , (i,j) \in \Omega
% 
% Author: Shuai Zhang 
% Date: April 2015
% version: lineartimesvd

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

function Out = TS1_s2(m,n,A,b,r,opts)

noise = 0; % no noise in default
if isfield(opts,'noise'),           noise = opts.noise;                end


rk = r;

Atb = AtMVMx(A,b,m,n);  % calculate AtMb

% initial value
% opts.x0 = zeros(m,n);
opts.x0 = Atb;



x = opts.x0; miu = opts.miu;
% intial value for lambda
% lambda = nrm2Atb*opts.eta; 
% lambda = nrm2Atb*min(3,1/opts.sr); 

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

while (i <= opts.mxitr) && (err >= opts.tol)
    itr_rank = itr_rank + 1;
    i = i + 1;
    
    xp = x;
    y = x - miu*g; % y = B_miu(x)
    
    if i == 1
        if max(m,n) < 1000
            [U,S,V] = svd(y); sigma = diag(S);
        else           
            [U,sigma] = LinearTimeSVD(y,sn,opts.maxr,pp); % v1
            
            invsigma = zeros(size(sigma)); indx = find(sigma);
            invsigma(indx) = 1./sigma(indx);
            V = (diag(invsigma)*U'*y)';
        end
        
    else
        % number of singular values to approximate
%         kk = length(find(sp > mx*opts.fastsvd_ratio_leading));
%         kk = max(r+1,min(kk,sn)); % kk >= r+1
        kk = min(r+3,sn);
        
%         [U,sigma] = LinearTimeSVD(y,sn,kk,pp);
        [U,sigma] = LinearTimeSVD(y,max(sn,kk+3),kk,pp); % v1
%         [U,S,~] = svds(y, kk); sigma = diag(S); % v2
%         [U,S,~] = rand_svd(y,kk,1,1,1); sigma = diag(S); % v3

    end
    
    s = sigma; % sigma is a vector
    
    if length(s) < r+1
        [U,S,~] = svds(y, r+1); sigma = diag(S); s = sigma;
    end
    
    if i > 1
        invsigma = zeros(size(sigma)); indx = find(sigma);
        invsigma(indx) = 1./sigma(indx);
        V = (diag(invsigma)*U'*y)';
    end
        
    ind = find(s > 0);
    if length(ind) > r
        Ue = U(:,ind); Ve = V(:,ind); s = s(ind);
    end
%----------------------------------------------
    if est_rank >= 1; rank_estimator_adaptive(); end
    if rk ~= r; r = rk; end
    
    
    
    %%  Adaptive Thresholding ...
    lambda = 2*s(r+1)^2/(1+2*s(r+1));
    a = lambda + sqrt(lambda^2+lambda*2);
    t = a/2;
    
    lambda = lambda/miu;
    idx = find ( s >= t);
    x = zeros(length(s),1); phi = x; 
    phi(idx) = acos( max(1 - 27/2*lambda*miu*(a+1)*a ./(a + s(idx)).^3,-0.99) );
    x(idx)= 2*(a+s(idx)).*cos(phi(idx)/3)/3 - 2*a/3 + s(idx)/3;
    
    S = diag(x); % S is a diagonal matrix
    x = Ue*S*Ve';
    s = diag(S);
    g = get_g(x,m,n,A,Atb);
    
    % check shrinkage
    nrmxxp = norm(x - xp,'fro');
    err = nrmxxp/max(norm(xp,'fro'),1);
    
    resid = norm(b-x(A)); % residual
    Objv(i) = resid/datanrm;
    

end



Out.rank = rank(x);
if ~noise;  x(A) = b; end % if no noise, add known information to improve accuracy
Out.x = x; 
Out.iter = i;
Out.obj = Objv(1:i);

% fprintf('Final value of para a is %e \n',a);

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
            rank_min = 1;
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