function evaluation_info=evaluate(XTrain,YTrain,XTest,YTest,LTest,LTrain,param)
    tic;
    
    [ B] = train(XTrain, YTrain, param, LTrain);
    theta = param.theta;
    yeta = param.yeta;


    Px = SAE(XTrain', B, theta, yeta);
    Py = SAE(YTrain', B, theta, yeta);


    
    
    fprintf('evaluating...\n');
    
  %% image as query to retrieve text database
    BxTest = compactbit(XTest*Px' >= 0);
    ByTrain = compactbit(B' >= 0);
    DHamm = hammingDist(BxTest, ByTrain);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Image_to_Text_MAP = mAP(orderH', LTrain, LTest)
    

    %% text as query to retrieve image database
    ByTest = compactbit(YTest*Py' >= 0);
    BxTrain = compactbit(B' >= 0);
    DHamm = hammingDist(ByTest, BxTrain);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Text_to_Image_MAP = mAP(orderH', LTrain, LTest)
      
end