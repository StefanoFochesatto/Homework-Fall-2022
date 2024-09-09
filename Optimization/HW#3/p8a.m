function z = p8a(f,df)
% This function takes a function f (doesn't really need it) 
% and it's derivative and attempts to find a local 
% minimum around with initial point x = 1 with a set 
% step size alphak.
x = 1;
alphak = .01;
while true
    %Stopping Criteria. 
    if abs(df(x)) < 1e-3
        break;
    else 
        %Descent direction finder. 
        if df(x) > 0
            pk = -1;
        else
            pk = 1;
        end
        %Applying descent direction and iterating. 
        x = x + alphak*pk;
     end
    
end
z = x;