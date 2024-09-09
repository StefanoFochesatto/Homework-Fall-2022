function x = phaseone(A,b)
    % Pull Sizes
    [m, n]= size(A);
    % Generate new phase one matrix
    A1 = [A eye(m)];
    % Generate phase one objective function
    c1 = [zeros(n, 1);ones(m, 1)];
    % Generate phase one feasible solution
    b1 = b(:);
    x1 = [zeros(n, 1);b];

    % Running simplex on phase one problem
    [x, z1] = sfsimplex(c1,A,b1,x1 ,true);

    % Reporting if problem is infeasible
    % no solution to Ax = b in the original problem
    if z1 > 0
        error('Infeasible')
    else
end