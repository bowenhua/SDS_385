function bigM = generateBigM(X,y, UB, idx)
%GENERATEBIGM Generate bounds for betas

M = size(X, 1);
N = size(X, 2);
OPTIONS = sdpsettings('solver','gurobi', 'verbose', 0);

beta = sdpvar(N, 1);

constraints = norm(y-X*beta,2) <= UB;

diagnostics = optimize(constraints,beta(idx),OPTIONS);
beta_plus = value(beta(idx));
diagnostics = optimize(constraints,-beta(idx),OPTIONS);
beta_minus = value(beta(idx));

bigM = max(abs(beta_plus), abs(beta_minus));

end

