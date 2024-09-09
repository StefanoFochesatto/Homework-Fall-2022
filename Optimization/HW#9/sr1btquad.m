function [xk, xklist] = sr1btquad(x0,Q,c,tol,maxiters)
% This function takes an initial value x0, a quadratic system 
% Q and c, a gradient tolerance, tol and maximum number of iter
% ations, maxiters. Performs symmetric rank-one quasi-Newton method 
% with exact line search. 

% Default number of maximum iterations. 
if nargin < 5
    maxiters = 20000;            
end
% Force column vectors
xk = x0(:);   

if nargout > 1
    xklist = [xk];
end

% Setting up first step 
n = length(xk);
B = eye(n,n);                    
fxk = .5*x0'*Q*x0 - c'*x0;
dfxk = Q*x0 - c;

for k = 1:maxiters
    % Tolerance Check
    if norm(dfxk) < tol          
        break
    end
    
    %Quasi-newton step with exact line search
    pk = - B \ dfxk;             
    if dfxk' * pk >= 0.0
        fprintf(['warning: non-descent direction' ...
            ' at step %d (use steepest-descent)\n'],k)
        pk = - dfxk;
    end
    % Exact line search
    alpha = (-pk'*dfxk)/(pk'*Q*pk);   
    % Saving old values for B computations.
    oldxk = xk;
    olddfxk = dfxk;
    % Update step. 
    xk = xk + alpha * pk;  
    % Append latest point to list.
    if nargout > 1
        xklist = [xklist xk];    
    end

    % Compute new function and gradient values
    fxk = .5*xk'*Q*xk - c'*xk;
    dfxk = Q*xk - c;

    % Update B
    sk = xk - oldxk;            
    yk = dfxk - olddfxk;
    v = yk - B * sk;
    
    denom = v' * sk;
    if denom == 0
        break
    end
    if abs(denom) < 1.0e-14
        warning(strcat('divide by near-zero in symmetric rank-one update',...
                       ' at iteration %d; convergence?'),k)
    end
    % symmetric rank-one update
    B = B + v * (v / denom)';   
    %disp(B)
end
end