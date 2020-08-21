clc;
clear all;
close all;
file1 = fopen('result_newton.txt','a+');
fprintf(file1,'%s\n',date);

max_trial =10;
global p1 p2;
% ep = 0.9;
            
for load_file = 1:1
    %% initializing variables
    no_part = 5.;
    %% to load file
    switch load_file

         case 1
     file = 'wpbc';
            test_start =131;
           
        otherwise
            continue;
    end
%parameters 
              cvs1=[10^-5,10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3,10^4,10^5];
%          cvs2=[10^-7,10^-6,10^-5,10^-4,10^-3,10^-2];
%         cvs1 = [2^-16,2^-14,2^-12,2^-10,2^-8,2^-6,2^-4,2^-2,2^0,2^2,2^4,2^6,2^8];
              uvs1=[0.3];
         
mus=[2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5];
              epsv=[0.5];

%Data file call from folder   
filename = strcat(file,'.txt');
    A = load(filename);
    [m,n] = size(A);
%define the class level +1 or -1    
    for i=1:m
        if A(i,n)==0
            A(i,n)=-1;
        end
    end
% Dividing the data in training and testing    
  [no_input,no_col] = size(A);
    test = A(test_start:m,:);
    train = A(1:test_start-1,:);
    x1 = train(:,1:no_col-1);
    y1 = train(:,no_col);
	    
    [no_test,no_col] = size(test);
    xtest0 = test(:,1:no_col-1);
    ytest0 = test(:,no_col);
% Normalize the data training and testing 
me=repmat(mean(x1),size(x1,1),1);
st=repmat(std(x1),size(x1,1),1);
tme=repmat(mean(x1),size(xtest0,1),1);
tst=repmat(std(x1),size(xtest0,1),1);

    x1 = (x1-me)./st;
    xtest0=(xtest0-tme)./tst;

    %% Universum
    A=[x1 y1;xtest0 ytest0];
    [no_input,no_col] = size(A);
   obs = A(:,no_col);   
%     C=A;
    C1= A(1:test_start-1,:);
    A = [];
 B = [];

for i = 1:test_start-1
    if(obs(i) == 1)
        A = [A;C1(i,1:no_col-1)];
    else
        B = [B;C1(i,1:no_col-1)];
    end;
end;

 u=ceil(uvs1*(test_start-1));
sb1=size(A,1);
sb=size(B,1);
ptb1=sb1/u;
ptb=sb/u;
Au=A(1:ptb1:sb1,:);
Bu=B(1:ptb:sb,:);
di=size(Au,1)-size(Bu,1);
if(di>0)
Bu=[Bu ;Bu(1:abs(di),:)];
elseif(di<0)
Au=[Au ;Au(1:abs(di),:)];
end   
 U=(Au+Bu)/2;   
  

    %Combining all the column in one variable
     A=[x1 y1];    %training data
    A_test=[xtest0,ytest0];    %testing data
 %% initializing crossvalidation variables

    [lengthA,n] = size(A);
    min_err = -10^-10.;
   

  for C1 = 1:length(cvs1)
            c = cvs1(C1)
            c2=c;
              for mui = 1:length(mus)
				p1 = mus(mui)
             
            for  ei = 1:length(epsv)
                    e = epsv(ei)
                    avgerror = 0;
                    block_size = lengthA/(no_part*1.0);
                    part = 0;
                    t_1 = 0;
                    t_2 = 0;
                    while ceil((part+1) * block_size) <= lengthA
                   %% seprating testing and training datapoints for
                   % crossvalidation
                                t_1 = ceil(part*block_size);
                                t_2 = ceil((part+1)*block_size);
                                B_t = [A(t_1+1 :t_2,:)];
                                Data = [A(1:t_1,:); A(t_2+1:lengthA,:)];
                   %% testing and training
                                [accuracy_with_zero,w,time] = gunnsvc(Data(:,1:no_col-1),Data(:,no_col),B_t(:,1:no_col-1),B_t(:,no_col),U,c,c2,e);
                                avgerror = avgerror + accuracy_with_zero;
                                part = part+1
                     end
       
           %% updating optimum c, L
          
           if avgerror > min_err
               min_err = avgerror;
               min_c1 = c;
                min_c2 = c2;
               min_e = e;
           end 
            end
            end
   end
 

 [accuracy,w,time,sen,spe,pre] = gunnsvc(A(:,1:no_col-1),A(:,no_col),A_test(:,1:no_col-1),A_test(:,no_col),U,min_c1,min_c2,min_e);

 fprintf(file1,'%s\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\tu=%g\tn=%g\n',file,size(A,1),size(A_test,1),accuracy,sen,spe,pre,min_c1,min_c2,min_e,time,min_u,n-1);

 
end
 