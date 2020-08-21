function [accuracy,predictedY,w,sen,spe,pre] = svcerror(trnX,trnY,tstX,tstY,ker,alpha,bias,U)
%SVCERROR Calculate SVC Error
%
%  Usage: err = svcerror(trnX,trnY,tstX,tstY,ker,alpha,bias)
%
%  Parameters: trnX   - Training inputs
%              trnY   - Training targets
%              tstX   - Test inputs
%              tstY   - Test targets
%              ker    - kernel function
%              beta   - Lagrange Multipliers
%              bias   - bias
%
%  Author: Steve Gunn (srg@ecs.soton.ac.uk)
% ueps=0.01;
[x y]=size(trnX);
[ux uy]=size(U);
y=ones(ux,1);
X1=[trnX;U;U];
Y1=[trnY;y;-y];
trnX=X1;
trnY=Y1;
  if (nargin ~= 8) % check correct number of arguments
    help svcerror
  else

    n = size(trnX,1);
    m = length(tstY);
    H = zeros(m,n);  
    for i=1:m
      for j=1:n
        H(i,j) = trnY(j)*svkernel(ker,tstX(i,:),trnX(j,:));
      end
    end
    predictedY = sign(H*alpha + bias);
    w=(alpha.*trnY)'*trnX;
    err = sum(predictedY ~= tstY);
    accuracy = (1-err/m)*100
    tp=sum(predictedY == tstY&predictedY == -1);
    tp=tp/10;
    fn=1-tp;
 
    tn=sum(predictedY == tstY&predictedY == 1);
    tn=tn/10;
    fp=1-tn;
    
    sen=tp*100/(tp+fn);
    spe=tn*100/(tn+fp);
    pre=tp*100/(tp+fp);
  end
