function [w, A, b] = total_lsqfast_GPU(samples, policy, new_policy, firsttime)

% total least squares modification of fixed point (TD) projection
% Sridhar Mahadevan, May 26 2013

% Key idea: normally, fixed point projection uses the projection 
% x = (A'A)^{-1}A'b 

% Total Least squares fixed point projection instead uses the projection 
% x = (A'A - \sigma_{n+1} I)^{-1} A'b 
% where \sigma_{n+1} is the n+1th singular value of [A, b] 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright 2000-2002
%
% Michail G. Lagoudakis (mgl@cs.duke.edu)
% Ronald Parr (parr@cs.duke.edu)
%
% Department of Computer Science
% Box 90129
% Duke University, NC 27708
%
% Copyright 2006
% Mauro Maggioni (mauro@math.duke.edu)
%
% [w, A, b] = lsqfast(samples, policy, new_policy, firsttime)
%
% Evaluates the "policy" using the set of "samples", that is, it
% learns a set of weights for the basis specified in new_policy to
% form the approximate Q-value of the "policy" and the improved
% "new_policy". The approximation is the fixed point of the Bellman
% equation.
%
% "firsttime" is a flag (0 or 1) to indicate whether this is the first
% time this set of samples is processed. Preprossesing of the set is
% triggered if "firstime"==1.
%
% Returns the learned weights w and the matrices A and b of the
% linear system Aw=b.
%
% See also lsq.m for a slower (incremental) implementation.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% VERBOSE=0; 

persistent Phihat;
persistent gPhihat; 
persistent Rhat;
persistent gRhat; 
persistent lastweights; 
%persistent sigmanplusone; % keep around the last singular value of Phihat 

%%% Initialize variables
howmany = length(samples);
k = feval(new_policy.basis);
% A = zeros(k, k);
% b = zeros(k, 1);
% PiPhihat = zeros(howmany,k);



%%% Precompute Phihat and Rhat for all subsequent iterations
if firsttime == 1
    Rhat   = cat(1,samples.reward);
    gRhat = gpuArray(Rhat); 
    lAllStateIdxs = 1:length(samples); lAllStateIdxs = lAllStateIdxs';
    Phihat = feval(new_policy.basis, lAllStateIdxs,cat(1,samples.action));
    gPhihat = gpuArray(Phihat); 
    lastweights = zeros(k,1); 
   
%     Phihat = sparse(Phihat);  % SRIDHAR: 10/26: Sparsify when graph is partitioned
end;

% Evaluate the basis functions at the states and actions in nextsample's
% Find the non absorbing states
lNotAbsorbIdxs = find(~cat(1,samples.absorb));
lNextStates = setdiff(lNotAbsorbIdxs,length(samples));
% Compute the policy at the next state
lNextActions = policy_function(policy,lNextStates+1);
% Evaluate the basis functions at the samples and corresponding actions
nextphi = feval(new_policy.basis, lNextStates+1, lNextActions);
PiPhihat = zeros(howmany,size(nextphi,1));
PiPhihat(lNextStates,:) = nextphi';
% PiPhihat = sparse(PiPhihat);  % SRIDHAR: 10/26: Sparsify when graph is partitioned. 

% Save some memory
clear lNextSamples lNextActions lNotAbsorbIdxs lAbsorb; %pack;



% need to compute the n+1th eigenvalue of [A; b] where
% A = Phihat*w and b = Rhat + discount*PiPhihat*w

% disp(size(Rhat));  disp(size(PiPhihat)); disp(size(lastweights)); 

% disp(lastweights'); 

bplus =  Rhat + new_policy.discount * PiPhihat * lastweights; 

[u_PhiR, s_PhiR, v_PhiR] = svd([Phihat',  bplus]); 

% use Tyger's Fast SVD algorithm 

%mytime = cputime;
%[u_PhiR, s_PhiR, v_PhiR] = pca([Phihat',  bplus], k+1); 
%svd_time = cputime - mytime;

svals_PhiR = diag(s_PhiR); 

%fprintf('CPU time to compute singular value %f using PCA: %f\n',svals_PhiR(k+1),svd_time); 

 %disp(svals_PhiR'); 

%    disp(svals_PhiR(k+1)); 
gsigmanplusonesquared = gpuArray(svals_PhiR(k+1)^2*eye(k)); 

%%% Compute the matrices A and b
gA = gPhihat * (gPhihat' - ggamma * gPiPhihat) - gsigmanplusonesquared; 
gb = gPhihat * gRhat;

gw = gA \ gb; 

w = gather(gw); 

%phi_time = cputime - mytime;
%if VERBOSE
%disp(['CPU time to form A and b : ' num2str(phi_time)]);
%end;
%mytime = cputime;

%%% Solve the system to find w
%if true, 
    % Standard inversion.
    % Use least squares \ or pinv depending on whether A is badly conditioned or not
    % TODO: one can gain a factor of 2 when pinv is used, since here both rank and pinv use the full svd
%     if issparse(A)
%         rankA = sprank(A); 
%     else rankA = rank(A); end; 
% 
%     rank_time = cputime - mytime;
%     disp(['CPU time to find the rank of A : ' num2str(phi_time)]);
%     mytime = cputime;
% 
%     disp(['Rank of matrix A : ' num2str(rankA)]);
%     if rankA==k
%         disp('A is full rank.');
%         w = A\b;
%     else
%         disp(['A is lower rank. Should be ' num2str(k)]);



%w = pinv(A)*b;

% gradually adapt weights 
beta = 0.9; 

lastweights = (1 - beta)*lastweights + beta*w; 

%     end

%    solve_time = cputime - mytime;
%if VERBOSE
%    disp(['CPU time to solve Aw=b : ' num2str(solve_time)]);
%end; 
%else
    % Multiscale inversion. The matrix T w.r.t which the multiscale structure is constructed is
    % new_policy.discount*(Phihat*Phihat')^{-1}*Phihat*PiPhihat
    % TODO: (Phihat*Phihat')^{-1}*Phihat does not change through iterations, so it could be computed once for all
    % This will not work when norm(Pnew)>1 (or, in practice, norm(Pnew)>~1.4, and there aren't more than 12 or so levels)    
%    B=Phihat*Phihat';Binv=pinv(B);    
%    Pnew=new_policy.discount*Binv*Phihat*PiPhihat;    
%    norm(Pnew),
    %figure;subplot(1,2,1);plot(log10(svd(B)));subplot(1,2,2);plot(log10(svd(Pnew)));
%    bnew=Binv*b;
%    w = DWSolveLinearSystem( Pnew, bnew, struct('Type','Normal','Precision',1e-6,'Reorthogonalize',true,'Threshold',1e-6,'IPThreshold',1e-12) );
    % For debug purposes only:
    % fprintf('Operator norm of Pnew:%f',norm(Pnew));   % Check if the norm of Pnew was too large...catastrophic effects will already been seen though!
%    w1 = pinv(eye(size(Pnew))-Pnew)*bnew;             % Solve with pinv
%    fprintf('Norm of difference between multiscale and svd solution: %f\n',norm(w-w1));   % Compare the solution of pinv with the one of DWSolveLinearSystem
%end;

return;
