function [nsv, alpha, b0,time] = svc(X,Y,ker,U,C,C1,ueps)
%SVC Support Vector Classification
%
%  Usage: [nsv alpha bias] = svc(X,Y,ker,C)
%
%  Parameters: X      - Training inputs
%              Y      - Training targets
%              ker    - kernel function
%              C      - upper bound (non-separable case)
%              nsv    - number of support vectors
%              alpha  - Lagrange Multipliers
%              b0     - bias term
%
%  Author: Steve Gunn (srg@ecs.soton.ac.uk)
% ueps=0.01;
[x y]=size(X);
[ux uy]=size(U);
y=ones(ux,1);
X1=[X;U;U];
Y1=[Y;y;-y];
X=X1;
Y=Y1;
C2=C1;
  if (nargin <2 | nargin>7) % check correct number of arguments
    help svc
  else

    fprintf('Support Vector Classification\n')
    fprintf('_____________________________\n')
    n = size(X,1);
    if (nargin<4) C=Inf;C2=Inf;, end
    if (nargin<3) ker='linear';, end

    % tolerance for Support Vector Detection
    epsilon = svtol(C);
    epsilon1 = svtol(C2);
    % Construct the Kernel matrix
    fprintf('Constructing ...\n');
    H = zeros(n,n);  
    tic
    for i=1:n
       for j=1:n
          H(i,j) = Y(i)*Y(j)*svkernel(ker,X(i,:),X(j,:));
       end
    end
    c1 = -ones(x,1);
    c2=ueps*ones(2*ux,1);
    c=[c1;c2];
    % Add small amount of zero order regularisation to 
    % avoid problems when Hessian is badly conditioned. 
    H = H+1e-10*eye(size(H));
    
    % Set up the parameters for the Optimisation problem

    vlb = zeros(n,1);      % Set the bounds: alphas >= 0
    vub1 = C*ones(x,1);     %                 alphas <= C
    vub=[vub1;C2*ones(2*ux,1)];
    x0 = zeros(n,1);       % The starting point is [0 0 0   0]
    neqcstr = nobias(ker); % Set the number of equality constraints (1 or 0)  
    if neqcstr
       A = Y';, b = 0;     % Set the constraint Ax = b
    else
       A = [];, b = [];
    end
%  A = Y';, b = 0;   
    % Solve the Optimisation Problem
    
    fprintf('Optimising ...\n');
%     st = cputime;
   
    [alpha lambda how] = quadprog(H,c,[],[],A,b,vlb,vub,x0);

    time = toc;
%     fprintf('Execution time: %4.1f seconds\n',cputime - st);
    fprintf('Status : %s\n',how);
    w2 = alpha'*H*alpha;
    fprintf('|w0|^2    : %f\n',w2);
    fprintf('Margin    : %f\n',2/sqrt(w2));
    fprintf('Sum alpha : %f\n',sum(alpha));
    
        
    % Compute the number of Support Vectors
    svi = find( alpha(1:x,:) > epsilon);
    svi1 = find( alpha(x+1:x+2*ux,:) > epsilon1);
    svi2=svi1+x;
    svi=[svi;svi2];
    nsv = length(svi);
   
    fprintf('Support Vectors : %d (%3.1f%%)\n',nsv,100*nsv/n);

    % Implicit bias, b0
    b0 = 0;

    % Explicit bias, b0 
    if nobias(ker) ~= 0
      % find b0 from average of support vectors on margin
      % SVs on margin have alphas: 0 < alpha < C
      svii = find( alpha(1:x,:) > epsilon & alpha(1:x,:) < (C - epsilon));
      svii1 = find( alpha(x+1:x+2*ux,:) > epsilon1 & alpha(x+1:x+2*ux,:) < (C2 - epsilon1));
      svii1=svii1+x;
      svii=[svii;svii1];
      
      if length(svii) > 0
		Y1=Y;
		Y1(x+1:x+2*ux)=-ueps*Y(x+1:x+2*ux);
		b0 =  (1/length(svii))*sum(Y1(svii) - H(svii,svi)*alpha(svi).*Y(svii));
       else 
        fprintf('No support vectors on margin - cannot compute bias.\n');
      end
    end
    
  end
 
    
