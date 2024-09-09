function [Hist,Val] = NewtonMethodMulti(f,g,xinit, xtol)
% This function takes a two function of the form 
% f = x + y + c where c is a constant, an initial value 
% xinit in the form of a column vector, and an xtolerance
% for computing Newton's method for a system of linear equations
% with two variables and two equations. 

%Symbolically computing the jacobian
syms x y 
Jacobian = jacobian([f,g],[x, y]);

%Evaluating the functions and jacobian
fx1 = double(subs(f, [x y], xinit'));
gx1 = double(subs(g, [x y], xinit'));
Jacobianx1 = double(subs(Jacobian, [x y], xinit'));

%First Newton step
x1 = xinit - inv(Jacobianx1)*[fx1;gx1]; % Column vector

Hist = [xinit x1]; % Hist each col is an iteration
Count =  2;


while(norm(Hist(:,Count) - Hist(:,(Count - 1))) >= xtol)
    xn = Hist(:, Count); % Assign current iterate
    
    % Compute function values and Jacobian
    fxn = double(subs(f, [x y], xn'));
    gxn = double(subs(g, [x y], xn'));
    Jacobianxn = double(subs(Jacobian, [x y], xn'));

    % Compute next iterate
    xn = xn - inv(Jacobianxn)*[fxn;gxn];

    % Append Hist iterate Count
    Hist = [Hist  xn];
    Count = Count + 1;
end

% Return final iterate
Val = Hist(:, Count);

% Sanity Check
fxfinal = double(subs(f, [x y], Val'))
gxfinal = double(subs(g, [x y], Val'))


end