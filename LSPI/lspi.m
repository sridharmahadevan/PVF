function [policy, all_policies] = lspi(domain, algorithm, maxiterations, ...
    epsilon, samples, basis, ...
    discount, initial_policy, rpi_opts)



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

%

% [policy, all_policies] = lspi(domain, algorithm, maxiterations,

%                               epsilon, samples, basis, discount,

%                               initial_policy)

%

% LSPI : Least-Squares Policy Iteration

%

% Finds a good policy given a set of samples and a basis

%

% Input:

%

% domain - A string containing the name of the domain

%          This string should be the prefix for all related

%          functions, for example if domain is 'chain' functions

%          should be chain_initialize_policy, chain_simulator,

%          etc.

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

%

% maxiterations - An integer indicating the maximum number of

%                 LSPI iterations

%

% epsilon - A small real number used as the termination

%           criterion. LSPI converges if the distance between

%           weights of consequtive iterations is less than

%           epsilon.

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

% basis - The function that computes the basis for a given pair

%         (state, action) given as a function handle

%         (e.g. @chain_phi) or as a string (e.g. 'chain_phi').

%

% discount - A real number in (0,1] to be used as the discount factor

%

% initial_policy - (optional argument) An initial policy for

%                  LSPI. It should be given as a struct with the

%                  following fields (at least):

%

%                  explore  : Exploration rate (real number)

%                  discount : Discount factor (real number)

%                  actions  : Total numbers of actions, |A|

%                  basis    : The function handle for the basis

%                             associated with this policy

%                  weights  : A column array of weights

%                             (one for each basis function)

%

%                  If initial_policy is not provided it is initialized

%                  to a policy with "explore"=0.0, "discount" and

%                  "basis" as provided above, and "actions" and

%                  "weights" as suggested by the domain (in the

%                  domain_initialize_policy function.

%

% Output:

%

% policy - The learned policy (same struct as above)

%

% all_policies - A cell array of size (iterations+1) containing

%                all the intermediate policies at each LSPI

%                iteration, including the initial policy.

%

% rpi_options - structure containing the following fields:

%               rpi_initializebasis_opts : structure to be passed to rpi_initialize_basis

%               rpi_basis_opts           : structure to be passed to the basis function contructor

%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

VERBOSE = 0; 




%%% Initialize the random number generator to a random state

rand('state', sum(100*clock));



%%% Create a new policy

initialize_policy = [domain '_initialize_policy'];

policy = feval(initialize_policy, 0.0, discount, basis);

%%% Consider this to be the initial_policy if one is not provided

if (nargin<8) | isempty(initial_policy),

    initial_policy = policy;

end



%%% Create the basis if necessary

initialize_basis = [domain '_initialize_basis'];

if nargin<9,

    rpi_opts = [];

end;

if ~isfield(rpi_opts,'rpi_initializebasis_opts'),

    rpi_opts.rpi_initializebasis_opts = [];

end;

if ~isfield(rpi_opts,'rpi_basis_opts'),

    rpi_opts.rpi_basis_opts = [];

end;

basis  = feval('rpi_initialize_basis',basis,samples,rpi_opts.rpi_initializebasis_opts,rpi_opts.rpi_basis_opts);



%%% Update the policy depending on the basis

initial_policy.k = feval(basis);

initial_policy.weights = zeros(initial_policy.k,1);



%%% Initialize policy iteration

iteration = 0;

distance = inf;

all_policies{1} = initial_policy;





%%% If no samples, return

if isempty(samples)

    disp('Warning: Empty sample set');

    return

end


gradient = inf; % measure improvement at each iteration 

%%% Main LSPI loop

while ( (iteration < maxiterations) && (distance > epsilon)) % && (gradient > epsilon)) 

    %%% Update and print the number of iterations

    iteration = iteration + 1;

   % disp('*********************************************************');
% 
%     disp( ['LSPI iteration : ', num2str(iteration)] );

    if (iteration==1)

        firsttime = 1;
        
        %gradient = inf; 

    else

        firsttime = 0;

    end


    %%% You can optionally make a call to collect_samples right here

    %%% to change/update the sample set. Make sure firsttime is set

    %%% to 1 if you do so.


    %%% Evaluate the current policy (and implicitly improve)

    %%% There are several options here - choose one

    if (algorithm == 1)
        
%        policy.weights = pnorm_rda_sparse_bdq(samples, all_policies{iteration}, policy,firsttime,iteration);
         policy.weights = larstd_batch(samples, all_policies{iteration}, policy,firsttime);
         
    elseif (algorithm == 2)
        
        %policy.weights = lsqfast_GPU(samples, all_policies{iteration}, policy, firsttime);
        policy.weights = lsqfast(samples, all_policies{iteration}, policy, firsttime);
        
    elseif (algorithm == 3)
        
%        policy.weights = total_larstd_batch(samples, all_policies{iteration}, policy,firsttime);
        policy.weights = LCP_TD_batch(samples, all_policies{iteration}, policy,firsttime);
        
    elseif (algorithm == 4)
        
   %     policy.weights = lsqbefast(samples, all_policies{iteration}, policy, firsttime);
%        policy.weights = qlambda_bdq_batch(samples, all_policies{iteration}, policy, firsttime);
    %   policy.weights = fast_sparse_qcd(samples, all_policies{iteration}, policy, firsttime);
%       policy.weights = greedy_tsqfast(samples, all_policies{iteration}, policy, firsttime);
       
       policy.weights = TDCfast_GPU(samples, all_policies{iteration}, policy, firsttime);
%       policy.weights = TDCfast(samples, all_policies{iteration}, policy, firsttime);
       % policy.weights = ROTDfast(samples, all_policies{iteration}, policy, firsttime);       
        
    elseif (algorithm == 5)
        
        %policy.weights = lsqfastKron(samples, all_policies{iteration}, policy, firsttime);
   %     policy.weights = comd_sparse_qlambda(samples, all_policies{iteration}, policy,firsttime,iteration);
          policy.weights = tsqfast(samples, all_policies{iteration}, policy, firsttime);
        
    elseif (algorithm == 6)
        
        policy.weights = bdq_batch(samples, all_policies{iteration}, policy, firsttime,iteration);

     elseif (algorithm == 7)
        
        %policy.weights = comd_batch(samples, all_policies{iteration}, policy, firsttime);
        %policy.weights = total_lsqbefast(samples, all_policies{iteration}, policy, firsttime);
%        policy.weights = total_bellman_res_min(samples, all_policies{iteration}, policy, firsttime);
        policy.weights = total_lsqfast_rqi(samples, all_policies{iteration}, policy, firsttime);
        
    elseif (algorithm == 8)
        
       % policy.weights = qlearning(samples, all_policies{iteration}, policy, firsttime,iteration);
        policy.weights = total_lsqfast_GPU(samples, all_policies{iteration}, policy, firsttime);
        
        
    end


% 
%     if ~firsttime,
% 
%         policy.weights(find(all_policies{iteration}.weights==0)) = 0;
% 
%     end;

    

%    jeff_pendulum_plotpol(policy,.1,.1);

%    jeff_mcar_plotpol(policy,.025,.002);





    %%% Compute the distance between the current and the previous policy

    l1 = length(policy.weights);

    l2 = length(all_policies{iteration}.weights);

    if (l1 == l2)

        difference = policy.weights - all_policies{iteration}.weights;

        LMAXnorm = norm(difference,inf);

        L2norm = norm(difference);

    else

        LMAXnorm = abs(norm(policy.weights,inf) - ...
            norm(all_policies{iteration}.weights,inf));

        L2norm = abs(norm(policy.weights) - ...
            norm(all_policies{iteration}.weights));

    end
    
    lmaxnorms(iteration+1) = LMAXnorm;
    l2norms(iteration+1) = L2norm;
%     
%     if iteration > 1
%         
%        gradient = abs(l2norms(iteration) - l2norms(iteration-1)); 
%        % fprintf('gradient: %f\n', gradient); 
%        
%     end; 
    

    distance = L2norm;

    %%% Print some information
    
%     if iteration==maxiterations
%         
%        fprintf('LSPI timed out in %d iterations\n', maxiterations); 
% 
%      disp( ['   Norms -> Lmax : ', num2str(LMAXnorm), ...
%          '   L2 : ',            num2str(L2norm)] );
%      
%    end; 

    %%% Store the current policy

    all_policies{iteration+1} = policy;





    %%% Depending on the domain, print additional info if needed

    feval([domain '_print_info'], all_policies, samples);



    %continue ; % skip the change of basis



    %%% Update the basis functions

%     try
% 
%         all_policies{iteration+1} = rpi_change_basis(basis,samples,policy,epsilon,policy.actions);
% 
%         fprintf('New basis size: %d\n',length(find(all_policies{iteration+1}.weights~=0)));
% 
%     catch
% 
%         % That's ok, there was no initialization function for this basis.
% 
%         fprintf('Warning: error in changing basis, or no change of basis available.\n');
% 
%     end;

end





%%% Display some info
% 
% disp('*********************************************************');
% 
% if (distance > epsilon)
% 
%     disp(['LSPI finished in ' num2str(iteration) ...
%         ' iterations WITHOUT CONVERGENCE to a fixed point']);
% 
% else
% 
%     disp(['LSPI converged in ' num2str(iteration) ' iterations']);
% 
% end
% 
% disp('********************************************************* ');

if VERBOSE

  figure(11); subplot(2,3,5); plot(l2norms, 'r'); hold on; plot(lmaxnorms, 'b'); legend('l2', 'lmax'); 
  drawnow; 

end; 



return

