function [final_policy, all_policies, samples] = rpi_learn(domain,maxiterations, epsilon, samples, maxepisodes, maxsteps, discount, basis, algorithm, policy, rpi_opts)

     VERBOSE=0; 
  
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
%
% Mauro Maggioni (mauro@math.duke.edu)
%
% Department of Mathematics
% Yale University
% 
%
% [final_policy, all_policies, samples] = rpi_learn(maxiterations, epsilon, samples, maxepisodes, maxsteps, discount, basis, algorithm, policy, rpi_opts)
%    
% Runs LSPI for a domain
%
% Input:
%
% domain    : Name of the domain
% maxiterations - An integer indicating the maximum number of
%                 LSPI iterations (default = 10)
% 
% epsilon - A small real number used as the termination
%           criterion. LSPI converges if the distance between
%           weights of consequtive iterations is less than
%           epsilon (default = 10^-5)
%
% samples - The sample set. This should be an array where each
%            entry samples(i) has the following form: 
%
%            samples(i).state     : Arbitrary description of state
%            samples(i).action    : An integer in [1,|A|]
%            samples(i).reward    : A real value
%            samples(i).nextstate : Arbitrary description
%            samples(i).absorb    : Absorbing nextstate? (0 or 1)
%
%            (default = [] - empty)
%
% maxepisodes - An integer indicating the maximum number of
%               episodes from which (additional) samples will be
%               collected.
%               (default = 1000, if samples is empty, 0 otherwise)
% 
% maxsteps - An integer indicating the maximum number of steps of each
%            episode (an episode may finish earlier if an absorbing
%            state is encountered). 
%            (default = 50)
%
% discount - A real number in (0,1] to be used as the discount factor
%            (default = 0.9)
%
% basis - The function that computes the basis for a given pair
%         (state, action) given as a function handle
%         (e.g. @pendulum_phi) or as a string (e.g. 'pendulum_phi')
%         (default = 'pendulum_basis_rbf_C')
% 
% algorithm - This is a number that indicates which evaluation
%             algorithm should be used (see the paper):
%
%             1-lsq       : The regular LSQ (incremental)
%             2-lsqfast   : A fast version of LSQ (uses more space)
%             3-lsqbe     : LSQ with Bellman error minimization 
%             4-lsqbefast : A fast version of LSQBE (more space)
%
%             LSQ is the evaluation algorithm for regular
%             LSPI. Use lsqfast in general, unless you have
%             really big sample sets. LSQBE is provided for
%             comparison purposes and completeness.
%             (default = 2)
%
% policy - (optional argument) A policy to be used for collecting the
%          (additional) samples and as the initial policy for LSPI. It
%          should be given as a struct with the following fields (at
%          least):
%
%          explore  : Exploration rate (real number)
%          discount : Discount factor (real number)
%          actions  : Total numbers of actions, |A|
%          basis    : The function handle for the basis
%                     associated with this policy
%          weights  : A column array of weights 
%                     (one for each basis function)
%
%          If a policy is not provided, samples will be collected by a
%          purely random policy initialized with "explore"=1.0,
%          "discount" and "basis" some dummy values, and "actions" and
%          "weights" as suggested by the pendulum domain (in the
%          pendulum_initialize_policy function. Notice that the
%          "basis" used by this policy can be different from the
%          "basis" above that is used for the LSPI iteration. 
%
% rpi_opts - structure containing the following fields:
%               rpi_initializebasis_opts : structure to be passed to rpi_initialize_basis
%               rpi_basis_opts           : structure to be passed to the basis function contructor
%
%
% Output:
%
% final_policy - The learned policy (same struct as above)
% 
% all_policies - A cell array of size (iterations+1) containing
%                all the intermediate policies at each LSPI
%                iteration, including the initial policy. 
%
% samples     - The set of all samples used for this run. Each entry
%               samples(i) has the following form:
%
%               samples(i).state     : Arbitrary description of state
%               samples(i).action    : An integer in [1,|A|]
%               samples(i).reward    : A real value
%               samples(i).nextstate : Arbitrary description
%               samples(i).absorb    : Absorbing nextstate? (0 or 1)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  %clear functions
  
  %%% Print some info
%  disp('*************************************************');
%  disp('RPI : Representation Policy Iteration');
%  disp('-------------------------------------------------');
  
%  disp(['Domain : ' domain]);
  
  if nargin < 1
    maxiterations = 10;
  end
%  disp(['Max RPI iterations : ' num2str(maxiterations)]);
  
  
  if nargin < 2
    epsilon = 10^(-5);
  end
%  disp(['Epsilon : ' num2str(epsilon)]);
  
  
  if nargin < 3
    samples = [];
  end
%  disp(['Samples in the initial set : ' num2str(length(samples))]);
  
  if nargin < 4
    if isempty(samples)
      maxepisodes = 1000;
    else
      maxepisodes = 0;
    end
  end
%  disp(['Episodes for sample collection : ' num2str(maxepisodes)]);

  
  
  if nargin < 5
    maxsteps = 50;
  end
%  disp(['Max steps in each episode : ' num2str(maxsteps)]);

  
  
  if nargin < 6
    discount = 0.9;
  end
%  disp(['Discount factor : ' num2str(discount)]);
  
  
  if nargin < 7
    basis = [domain,'_basis_eigen'];
  end
%  disp('Basis : ');
%  disp(basis);
  
  
  %%% Set the evaluation algorithm
  if (nargin < 8) | (isempty(algorithm)),
    algorithm = 2;
  end
%   algorithms = ['lsq      '; 'lsqfast  '; 'lsqbe    '; 'lsqbefast'; 'lsq'];
%   disp(['Selected evaluation algorithm : ' algorithms(algorithm,: )]);

  if (nargin < 10),
      rpi_opts = [];
  end;
  
  %%% Collect (additional) samples if requested
  if maxepisodes>0
%     disp('-------------------------------------------------');
%     disp('Collecting samples ...');
    if (nargin < 9) | (isempty(policy)),
%       disp('... using a purely random policy');
      new_samples = collect_samples(domain, maxepisodes, maxsteps);
    else
%       disp('... using the policy provided');
      new_samples = collect_samples(domain, maxepisodes, maxsteps, policy);
    end
    samples = [samples new_samples];
    clear new_samples;
  end
  
  if (nargin < 9) | (isempty(policy)),
%    disp('No initial policy provided');
    policy = pendulum_initialize_policy(0.0, discount, basis);
  else
%    disp(['Initial policy exploration rate : ' num2str(policy.explore)]);
%    disp('Initial policy basis : ');
%    disp(policy.basis);
%    disp(['Initial policy number of weights : ', ...
%	  num2str(length(policy.weights))]);
  end  
  
  %%% Ready to go - Display the total number of samples
%   disp('-------------------------------------------------');
%   disp(['Total number of samples : ' num2str(length(samples))]);
    
  
  %%% Run LSPI
%  disp('*************************************************');
%  disp('Starting RPI ...');
  [final_policy, all_policies] = lspi(domain, algorithm, maxiterations, ...
				      epsilon, samples, basis, ...
				      discount, [], rpi_opts);
