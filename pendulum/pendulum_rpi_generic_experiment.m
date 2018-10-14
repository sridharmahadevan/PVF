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
% A script that runs the following experiment "rep" times,
% averages the results, plots them, and saves them in "experiment1":
%
%    1. Collect training data from n=[0:epistp:maxepi] episodes 
%    2. Run LSPI on those data until convergence or max 15 iterations
%    3. Evaluate the resulting policy in each case (pendulum_evalpol)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function all_its = pendulum_rpi_generic_experiment(numbases, rep, algm, basis) 

epi = 5:5:500;

max_steps = 30; 

best_steps = 0; 

discount = 0.9;

success_steps = 1000;

% init parallel

% workers = 10;
% 
% %Init parallel if the parallel toolbox is installed
% if exist('matlabpool') > 1 %#ok<EXIST>
%     if (matlabpool('size') == 0), matlabpool('open', 'local', workers); end
% end

all_its = zeros(rep,length(epi)); 

parfor test = 1:rep
    its = pendulum_rpi_generic(test, epi, max_steps, best_steps, discount, success_steps, basis,numbases,algm);
    all_its(test,:) = its;
    fprintf('test %d completed\n', test); 
end

figure; 
errorbar(mean(all_its),std(all_its));
xlabel('episode number');
ylabel('Average steps to completion');
title(sprintf('LSPI Algorithm %d on Inverted Pendulum Task over %d runs',algm, rep));
grid on; 



