function [success_prob, avesteps, avesteps_ebs, minsteps, maxsteps, avedrews, avedrews_ebs, steps, drews, urews] = pendulum_evalpol_graphics(pol, howmany, maxsteps)
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright 2000-2002 
%
% Michail G. Lagoudakis (mgl@cs.duke.edu)
% Ronald Parr (parr@cs.duke.edu)
%
% Department of Computer Science
% Box 90129
% Duke University
% Durham, NC 27708
% 
%
% [success_prob, avesteps, avesteps_ebs, minsteps, maxsteps, ...
%	  avedrews, avedrews_ebs, steps, drews, urews] = ...
%    pendulum_evalpol(pol, howmany, maxsteps)
%
% Evaluates the policy "pol" by running "howmany" episodes of
% "maxsteps" each. It then returns the statistics shown above (ebs
% stands for the 95% confidence intervals). A successful episode is
% one that reaches the maxsteps.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%   disp('Evaluating policy:');
%   disp(pol);
%   
  steps = zeros(1,howmany);
  drews = zeros(1,howmany);
  urews = zeros(1,howmany);
  
  for i=1:howmany
    
    state = pendulum_simulator;
    [steps(i), drews(i), urews(i)] = justexec(state, 'pendulum_simulator', ...
					      pol, maxsteps);
    
  end
  
  success_prob = length(find(steps==maxsteps)) / howmany; 
  
  avesteps = mean(steps);
  avesteps_std = std(steps,0,2);
  avesteps_ebs = 1.96 * avesteps_std ./ sqrt( size(steps,2) );
  
  minsteps = min(steps);
  maxsteps = max(steps); 
  
  avedrews = mean(drews);
  avedrews_std = std(drews,0,2);
  avedrews_ebs = 1.96 * avedrews_std ./ sqrt( size(drews,2) );
  
%   disp(['   Probability of success : ' num2str(success_prob)]);
%   disp(['   Average number of steps: ' num2str(avesteps) ' +/- ' ...
% 	num2str(avesteps_ebs)]);
%   disp(['   Minimum / Maximum steps: [' num2str(minsteps) ' ' ...
% 	num2str(maxsteps) ']']);
%   disp(['   Average total d. reward: ' num2str(avedrews) ' +/- ' ...
% 	num2str(avedrews_ebs)]);
  
  return
