function [allstep1] = pendulum_rpi_generic(rep, epi,max_steps, best_steps, discount, success_steps, basis,numbases, algm)

howmany = 1;

flag = true;   % recompute basis only when needed 

for i=1:length(epi)
    
%     disp('*****************************'); 
%     fprintf('Starting trial %d episode %d\n', rep, epi(i)); 

    if i==1
        numepi = epi(i);
        sam = [];
    else
        numepi = epi(i) - epi(i-1);
    end
    
    cnt = 0;    att=0;
    
    while (cnt < numepi)
        att = att + 1;
%        if i==1
%        else new_samples =  collect_samples('pendulum', 1, max_steps,allpol1(r,i-1));
%        end; 
        sam = [sam collect_samples('pendulum', 1, max_steps)];
        cnt = cnt + 1;
%        clear new_samples;
    end
    
   [allpol1(i), ~, sam] = rpi_learn('pendulum',10,10^-2,sam,...
     0,0,discount,basis,2,[],...
     struct('rpi_initializebasis_opts',...
     struct('SizeRandomSubset',1000),'rpi_basis_opts',...
     struct('NormalizationType','graph','Type','nn',...
     'BlockSize',10,'MaxEigenVals',50,'Delta',1.5,'DownsampleDelta',...
     0.1,'NNsymm','ave','kNN', 25, 'Rescaling', [3 1])));
    
    
    
%     while (cnt < numepi)
%         att = att + 1;
%         if i==1
%             new_samples = collect_samples('pendulum', 1, max_steps);
%         elseif flag==true new_samples =  collect_samples('pendulum', 1, max_steps,best_policy);
%         else new_samples = collect_samples('pendulum', 1, max_steps);
%         end;
%         sam = [sam new_samples];
%         cnt = cnt + 1;
%         clear new_samples;
%     end
%     
%     [allpol1(i), ~, sam] = rpi_learn('pendulum',50,10^-5,sam,...
%         0,0,discount,basis,algm,[],...
%         struct('rpi_initializebasis_opts',...
%         struct('SizeRandomSubset',1000),'rpi_basis_opts',...
%         struct('NormalizationType','graphmarkov','Type','nn',...
%         'BlockSize',10,'MaxFunctions', numbases,'MaxEigenVals',numbases,'Delta',1.5,'DownsampleDelta',...
%         0.1,'NNsymm','ave','kNN', 25, 'Rescaling', [3 1])));
    
    [allprob1(i), allstep1(i)] = pendulum_evalpol_graphics(allpol1(i), ...
        howmany, success_steps);
    
    fprintf('Solution: %d at run %d episode %d\n', allstep1(i), rep, epi(i)); 
    
    if i==1
        best_policy = allpol1(i);
        best_steps = allstep1(i);
    else if allstep1(i) > best_steps
            flag=true;
            best_steps = allstep1(i);
            best_policy = allpol1(i);
        else flag = false;
        end;
    end;
    
        
end




