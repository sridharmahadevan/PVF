

% generic comparison routine between two algorithms


function [all_its_algmA all_its_algmB] = pendulum_rpi_comparison_generic(numbases, rep, basis, algmA, algmB)



all_its_algmA = pendulum_rpi_generic_experiment(numbases, rep, algmA, basis); 

all_its_algmB = pendulum_rpi_generic_experiment(numbases, rep, algmB, basis); 


figure; 
plot(mean(all_its_algmA), 'r-*'); 

hold on; plot(mean(all_its_algmB), 'b-*'); 

grid on;

legend(num2str(algmA), num2str(algmB)); 

% figure; 
% 
% errorbar(mean(all_its_algmA), std(all_its_algmB)); 
% 
% title(num2str(algmA)); 
% 
% figure; 
% 
% errorbar(mean(all_its_algmA), std(all_its_algmB)); 
% 
% title(num2str(algmB)); 
% 
% grid on;