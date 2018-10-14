
function [weights, active] = bdq_batch(samples, policy, new_policy, firsttime,iteration)

%Phihat, Rhat, PiPhihat, discount,params)


% Sridhar Mahadevan, September 10 2011
% iterative computation of L1 regularized fixed point using coordinate
% descent
% use GLMNET as a subroutine


%%
% IN:
%
% OUT:
%   weights  :  Px1 vector of basis function weights where P<=K
%    active  :  Px1 vector of active basis functions s.t. Qhat = Phihat(active,:)'*weights
%   regpath  :  regpath.weights  :  KxnumSteps matrix of weights for each step of LARS-TD
%               regpath.beta     :  1xnumSteps vector of betaBar values for each LARS-TD step
%

persistent Phihat;
persistent Rhat;

VERBOSE = 0; 

lambda = 0.01; 

%%% Initialize variables
howmany = length(samples);
%   k = feval(new_policy.basis);

if firsttime==1
    
    %%% Precompute Phihat and Rhat for all subsequent iterations
    
    Rhat   = cat(1,samples.reward);
    lAllStateIdxs = 1:length(samples); lAllStateIdxs = lAllStateIdxs';
    Phihat = feval(new_policy.basis, lAllStateIdxs,cat(1,samples.action));
    
end;

% Evaluate the basis functions at the states and actions in nextsample's
% Find the non absorbing states
lNotAbsorbIdxs = find(~cat(1,samples.absorb));
lNextStates = setdiff(lNotAbsorbIdxs,length(samples));
% Compute the policy at the next state
lNextActions = policy_function(policy,lNextStates+1);
% Evaluate the basis functions at the samples and corresponding actions
nextphi = feval(new_policy.basis, lNextStates+1, lNextActions);
% PiPhihat = zeros(howmany,size(nextphi,1));

PiPhihat = zeros(size(nextphi,1),howmany); 
PiPhihat(:,lNextStates) = nextphi;
% PiPhihat = sparse(PiPhihat);  % SRIDHAR: 10/26: Sparsify when graph is partitioned.


%%%% Sizes
[K, M] = size(Phihat);

%%%% Initialization
w = zeros(K, 1);           % weights are 0

%%%% Main Loop
breakAlgo = false;

MaxIters = 10; % floor(howmany/100);  % replay increases as more data is collected 

errlist = [];

iters = 0;

bdgradient = inf; 

firsttime=1;

%     fprintf('finding fixpoint using coordinate descent \n');

pnormlist = []; 
qnormlist = []; 

while ~breakAlgo
    
    iters = iters + 1;
    
  %  alpha = 0.9/sqrt(iters);
  
    alpha = 1e-3/sqrt(iters);
    
    % parameters for Bregman Divergence Q-learning
      PNORM = max(floor(2*log10(K))/power(iters,2),2);
     
%     PNORM = 2; 
    
   % PNORM = 2*log10(K); 
    
    %PNORM = max(2*log(k)/sqrt(iteration),2);
    
    QNORM = PNORM/(PNORM-1);
    
    pnormlist = [pnormlist, PNORM]; 
    qnormlist = [qnormlist, QNORM]; 

    
    % [newweights, active, mscd] =  sparse_bdq2(Phihat,Rhat,PiPhihat, new_policy.discount,PNORM,QNORM,alpha, lambda,firsttime);
    
    [newweights, active, mscd] =  sparse_bdq2(Phihat,Rhat,PiPhihat, new_policy.discount,PNORM,QNORM,alpha, lambda,firsttime);
    
    if firsttime
        firsttime = 0;
    end;
    
    oldweights = w;
    
    w(active) = newweights;
    
    w(~active) = 0;
    
    err = norm(w-oldweights);
    
    errlist = [errlist, err];
    
    if iters>1
        bdgradient = abs(errlist(iters) - errlist(iters-1)); 
    end; 
    
    
    if (err <= 1e-2) || (iters > MaxIters ) || bdgradient < 1e-2 
        %(err <= 1e-2) || (iters > MaxIters )
        breakAlgo = 1;
    end;
    
    
end;

active = find(abs(w) > 0);
weights = w; % w(active);  % since lspi.m expects to see all weights 

if VERBOSE 

    figure(11); subplot(2,3,1); plot(errlist); hold on; title('BDQ');
    subplot(2,3,2); plot(w); str = ['total weights: ', num2str(K), ' active: ', num2str(length(active))];
    title(str);

    c = Phihat*(Rhat  + new_policy.discount*PiPhihat'*w - Phihat'*w);
    subplot(2,3,3); plot(c); title('correlations');

    subplot(2,3,4); plot(pnormlist, 'r'); hold on; plot(qnormlist, 'b'); legend('PNORM', 'QNORM'); 
end; 

return;

        
      