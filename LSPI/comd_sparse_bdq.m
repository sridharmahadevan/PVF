function [weights] = comd_sparse_bdq(samples, policy, new_policy, firsttime,iteration)

%%%%%%%%%%%%%%%%%
% Sridhar Mahadevan: sparse Qlearning based on Bregman Divergence
% this uses the adaptive mirror descent method following Duchi et al, JMLR 
%
% October 2011 
%%%%%%%%%%%%%%%%%%%%%%

% persistent g;  % keep track of gradients 

alpha = 0.9/sqrt(iteration); 

lambda = 1e-6; 

%%% Initialize variables
howmany = length(samples); 
k = feval(policy.basis);

if firsttime==1
    w = ones(k,1);
    
else w = policy.weights;
end;

% g = 1e-3*rand(k,k); 

epsilon = 1e-3; 

g = speye(k)*epsilon; 

for i=1:howmany
    
    %%% Compute the basis for the current state and action
     phi = feval(new_policy.basis, samples(i).state, samples(i).action);
     
      if ~samples(i).absorb
      
      %%% Compute the policy and the corresponding basis at the next state 
      nextaction = policy_function(policy, samples(i).nextstate);
      nextphi = feval(new_policy.basis, samples(i).nextstate, nextaction);
      
   %   figure(12); plot(nextphi); pause; 
      
    else
      nextphi = zeros(k, 1);
    end
    
    % find TD error
    td_error = samples(i).reward + new_policy.discount*(nextphi'*w) - phi'*w;
    
    % update gradient matrix 
    
    grad_vec = td_error*phi; 
    
    g = g + grad_vec*grad_vec';
    
    sqrt_g  = sqrt(diag(g));
    
    for j=1:k
        
        gterm =  w(j) + alpha*grad_vec(j)/sqrt_g(j);
        
        w(j) = sign(gterm)*max(0,abs(gterm) + lambda*alpha/sqrt_g(j));
        
    end;
    
    
end


figure(11); subplot(2,2,3); plot(sqrt_g);

weights = w;
