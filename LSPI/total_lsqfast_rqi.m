function [w, A, b] = total_lsqfast_rqi(samples, policy, new_policy, firsttime)

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
persistent Rhat;

%%% Initialize variables
howmany = length(samples);
k = feval(new_policy.basis);
% A = zeros(k, k);
% b = zeros(k, 1);
% PiPhihat = zeros(howmany,k);

%%% Precompute Phihat and Rhat for all subsequent iterations
if firsttime == 1
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
PiPhihat = zeros(howmany,size(nextphi,1));
PiPhihat(lNextStates,:) = nextphi';

% Save some memory
clear lNextSamples lNextActions lNotAbsorbIdxs lAbsorb; %pack;

A = Phihat*Phihat' - new_policy.discount * Phihat * PiPhihat; 

b = Phihat * Rhat; 

w = rqi(A, b); 

return;
