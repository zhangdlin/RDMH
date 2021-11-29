function [ B] = train(X1, X2, param, L)

fprintf('training...\n');

%% set the parameters
nbits = param.nbits;
alphaX1 = param.lambdaX;  
alphaX2 = 1-alphaX1;    
beide = param.beide;
lambda = param.lambda; 
run = 3;
times = 10;



%% get the dimensions
[n, dX] = size(X1);
dY = size(X2,2);
[~, ly] = size(L);

%% transpose the matrices
X1 = X1'; X2 = X2'; L = L';

%% initialization
B = sign(randn(nbits, n));
U1 = randn(dX ,nbits);
U2 = randn(dY ,nbits);

%% iterative optimization
for iter = 1:param.iter1
    
%     % update D1 
      itervalue_d1 = zeros(size(X1,2),1);
    for iterd1 = 1:size(X1,2)
     itervalue_d1(iterd1) = 1/(norm((X1(:,iterd1)-U1*B(:,iterd1)),2)+1e-5);
    end
        D1 = diag(itervalue_d1);
%% update D2 
    itervalue_d2 = zeros(size(X2,2),1);
     for iterd2 = 1:size(X2,2)
      itervalue_d2(iterd2) = 1/(norm((X2(:,iterd2)-U2*B(:,iterd2)),2)+1e-5);
     end
            D2 = diag(itervalue_d2);
 %% update U1
   U1 = alphaX1*X1*D1*B'/(alphaX1*B*D1*B'+lambda*eye(nbits));
  % update U2
   U2 = alphaX2*X2*D2*B'/(alphaX2*B*D2*B'+lambda*eye(nbits));
  % update P
   P = beide*B*L'/(beide*L*L'+lambda*eye(ly));
   Q1 = alphaX1*D1*X1'*U1 +0.5* beide*L'*P' ;
   Q2 = alphaX2*D2*X2'*U2 +0.5* beide*L'*P' ;
   for test = 1: run
       for t = 1: times
            B0 = B;
            rnd = size(B, 1);
            for i = 1: randperm(rnd)
                     
                    q1 = Q1(:,i); 
                    q2 = Q2(:,i);
                    u1 = U1(:,i); U1_ =U1; U1_(:,i) = [];
                    u2 = U2(:,i); U2_ =U2; U2_(:,i) = [];
                    b = B(i, :); B_ = B; B_(i,:)= [];
                    dx1 = itervalue_d1(i);
                    dx2 = itervalue_d2(i); 
                    B(i,:) = sign((q1'+alphaX1*(-u1'*U1_*B_*D1-0.5*alphaX1*u1'*u1*dx1))+(q2'+alphaX2*(-u2'*U2_*B_*D2-0.5*alphaX2*u2'*u2*dx2)));       
                
            end
            if norm(B0 - B, 'fro') < 1e-6 * norm(B0, 'fro')
                    break;
            end
       end
        temp{test} = B;
   end
        B = sign(temp{1} + temp{2} + temp{3});
        clear temp


      fprintf('\nobj at iiteration %d\n', iter);


end
return;
end

