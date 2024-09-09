function [xk, xklist] = sr1btfd(x0,f,tol,maxiters)
% This code is a modification of SR1BT 
% only requires the user to provide an inline function f,
% initial value x0, gradient tolerance tol, and maximum
% iterations maxiters. For computing the gradient it uses
% the finite difference formula. 

if nargin < 4
    maxiters = 20000;            % never take more steps than this
end
xk = x0(:);                      % force into column shape
if nargout > 1
    xklist = [xk];
end
n = length(xk);
h = sqrt(eps);

B = eye(n,n);                    % first step is steepest-descent: B=I
fxk = f(xk);
dfxk = finitedifference(xk, f, h);


for k = 1:maxiters
    if norm(dfxk) < tol          % absolute tolerance on gradient f
        break
    end
    pk = - B \ dfxk;             % quasi-Newton step
    if dfxk' * pk >= 0.0
        fprintf('warning: non-descent direction at step %d (use steepest-descent)\n',k)
        pk = - dfxk;
    end
    alpha = bt(xk,pk,dfxk,f);    % back-tracking line search
    oldxk = xk;
    olddfxk = dfxk;
    xk = xk + alpha * pk;        % do update
    if nargout > 1
        xklist = [xklist xk];    % append latest point to list
    end
    fxk = f(xk);
    dfxk = finitedifference(xk, f, h);
    sk = xk - oldxk;             % sk, yk used to update B; see GNS p. 413
    yk = dfxk - olddfxk;
    v = yk - B * sk;
    denom = v' * sk;
    if denom == 0
        break                    % in this case we are certainly done
    end
    if abs(denom) < 1.0e-14
        warning(strcat('divide by near-zero in symmetric rank-one update',...
                       ' at iteration %d; convergence?'),k)
    end
    B = B + v * (v / denom)';    % symmetric rank-one update
end
end % function

    function alpha = bt(xk,pk,dfxk,f)
    % BT Apply backtracking using standard default parameters.
    Dk = dfxk' * pk;
    fxk = f(xk);
    mu = 1.0e-4;  % modest sufficient decrease
    rho = 0.5;    % backtracking by halving
    alpha = 1.0;  % alpha_0=1 because of Newton, and by Thm 11.7
    while f(xk + alpha * pk) > fxk + mu * alpha * Dk
        alpha = rho * alpha;
    end
end % function

% Finite Difference Function
    function df = finitedifference(xk,f,h)
        n = length(xk);
        df = zeros(n,1);
        for i = 1:n
            xkh = xk;
            xkh(i) = xkh(i) + h;
            df(i) = (f(xkh) - f(xk))/h;
        end    
end