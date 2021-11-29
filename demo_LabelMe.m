
% Please note that the demo applies only to LabelMe dataset!!!
% If you use other datasets, please select the corresponding parameters!!!

%  If you have any questions, please contact me at anytime(dlinzzhang@gmail.com).
% If you use our code, please cite our article "Robust and discrete matrix factorization hashing for cross-modal retrieval".

function demo_LabelMe()
clc
clear all
% nbits_set=[16 32 64];
nbits_set=[32];


%% load dataset
load('dataset-only labelMe.mat');


XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
XTest = I_te; YTest = T_te; LTest = L_te;


%% initialization
% fprintf('initializing...\n')
param.lambdaX = 0.5; %
param.beide = 1; % 
param.lambda = 10;  % 
param.theta = 1e-6; % 
param.yeta = 0.001; %
param.iter1 = 20;
run = 1;


%% centralization
fprintf('centralizing data...\n');
XTest = bsxfun(@minus, XTest, mean(XTrain, 1)); XTrain = bsxfun(@minus, XTrain, mean(XTrain, 1));
YTest = bsxfun(@minus, YTest, mean(YTrain, 1)); YTrain = bsxfun(@minus, YTrain, mean(YTrain, 1));

%% kernelization
fprintf('kernelizing...\n\n');
[XKTrain,XKTest]=Kernelize(XTrain,XTest); [YKTrain,YKTest]=Kernelize(YTrain,YTest);
XKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1)); XKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));
YKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1)); YKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));

% %% evaluation

for bit=1:length(nbits_set) 
    nbits=nbits_set(bit);
      param.h = nbits;
    
         for rr = 1:run
    %% RDMH
    param.nbits=nbits;
    eva_info =evaluate(XKTrain,YKTrain,XKTest,YKTest,LTest,LTrain,param);
    
    
    % MAP
    Image_to_Text_MAP = eva_info.Image_to_Text_MAP;
    Text_to_Image_MAP=eva_info.Text_to_Image_MAP;
    
    fprintf('RDMH %d bits --  Image_to_Text_MAP: %f ; Text_to_Image_MAP: %f /n;',nbits,Image_to_Text_MAP,Text_to_Image_MAP);
    map(rr,1) = Image_to_Text_MAP;
    map(rr,2) = Text_to_Image_MAP;
 
    
         end 
   % Image_to_Text_MAP = mean(map(:,1));
   % Text_to_Image_MAP = mean(map(:,2));
        
end
fprintf('Average Map Img2Txt %d runs: %.4f\n', run, mean(map(:,1)));
fprintf('Average Map Txt2Img %d runs: %.4f\n', run, mean(map(:,2)));

