function pendulum_print_info(all_policies,cSamples)
  
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
% pendulum_print_info(all_policies)
%
% Calls several functions to compute/display info about the last policy
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

last = length(all_policies);
%pendulum_plotpol(all_policies{last}, 0.2, 0.5);

% pendulum_plotval(all_policies{last}, 0.1, 0, last-1);
%   pendulum_plotval23d(all_policies{last}, 0.1, 0.1, 0);

  %pendulum_test(all_policies{last}, 1000, last-1);
%  pendulum_evalpol(all_policies{last}, 100, 1800);
  
  
  % Uncomment to save the policies so far in the file "allpolfile"
  %save allpolfile all_policies  
  
  return
