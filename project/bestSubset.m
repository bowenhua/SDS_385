function [beta, obj] = bestSubset(X,y, bigM,bigM_L1, k)
%BESTSUBSET Best Subset Selection problem from Bertsimas(2016)
M = size(X, 1);
N = size(X, 2);
OPTIONS = sdpsettings('gurobi.MIPGap',0.01,'gurobi.TimeLimit', 100000, 'solver','gurobi', 'verbose', 2);

beta = sdpvar(N, 1);
z = binvar(N-1, 1);

% We don't penalize the intercept
constraints = -bigM(1:N-1).*z <= beta(1:N-1) <= bigM(1:N-1) .* z; 
constraints = [constraints, sum(z) <= k];
constraints = [constraints, -bigM <= beta <= bigM];
constraints = [constraints, norm(beta(1:N-1),1) <= bigM_L1];
obj = norm(y-X*beta,2);

diagnostics = optimize(constraints,obj,OPTIONS);

beta = value(beta);
obj = value(obj);
end

