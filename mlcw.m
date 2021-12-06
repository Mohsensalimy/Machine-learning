clear;clc

%% Load and reshape data -- https://www.mathworks.com/help/deeplearning/ug/data-sets-for-deep-learning.html

old=  addpath(fullfile(...
            matlabroot,'examples' , 'nnet' , 'main'));
xtr=processImagesMNIST('train-images-idx3-ubyte.gz') ;%squeeze(...extractdata(...
ytr=processLabelsMNIST('train-labels-idx1-ubyte.gz') ;
xte=processImagesMNIST( 't10k-images-idx3-ubyte.gz') ;
yte=processLabelsMNIST( 't10k-labels-idx1-ubyte.gz') ;path(old)

disp('Reshaping data')

Xtr=zeros(     size(xtr,  4),...
          prod(size(xtr,1:2)));
Xte=zeros(     size(xte,  4),...
          prod(size(xte,1:2)));
for   i=1: max(size(Xtr,1),...
               size(Xte,1))
         if i<=size(Xtr,1)
                    Xtr(i,:)=reshape(xtr(:,:,1,i),size(...
                    Xtr(i,:)));
         end
         if i<=size(Xte,1)
                    Xte(i,:)=reshape(xte(:,:,1,i),size(...
                    Xte(i,:)));
         end
end

%% Fit data

%  Random forest
disp('Training random forest')
%  From https://www.mathworks.com/help/stats/fitcensemble.html:
%        If 'Method' is 'Bag', then fitcensemble uses bagging with random predictor selections at each split (random forest) by default
%  Also:
%       For reproducibility of random forest algorithm, specify the 'Reproducible' name-value pair argument as true for tree learners
 r= fitcensemble(Xtr,ytr,'Method','Bag',...
                       'Learners',templateTree('Reproducible',true));
yr=      predict(  r,Xte);

%  Decision tree
disp('Training classification (decision) tree')
%  Limited MaxNumSplits, to avoid a 'messy' tree plot
 t= fitctree    (Xtr,ytr,'MaxNumSplits', 9e0 , 'Reproducible',true) ;
            view(  t,      'Mode','graph');
yn=      predict(  t,Xte);

save m0806
%% Heatmap for each classfiction 


%% 



RFcm = confusionmat(yte,yn); % random forest confusion matrix
figure(2)
movegui('west')
confusionchart(RFcm);
title('Random forest Confusion Matrix')
%% DT 


DTcm = confusionmat(yte,yr); % DT confusion matrix
figure(2)
movegui('west')
confusionchart(DTcm);
title('DTConfusion Matrix')

%%  DT test error 
DTtestErr = loss(t,Xte,yte);
disp("DT Test error: " + DTtestErr)

%% %% Test error Random forest
RFtestErr = loss(r,Xte,yte);
disp("RF Test error: " + RFtestErr)

%% 
%% DT train error
DTtrainErr = loss(t,Xtr,ytr);
disp("DT train error: " + DTtrainErr)

%% %% Train error Random forest
RFtrainErr = loss(r,Xtr,ytr);
disp("RF train error: " + RFtrainErr)

%% DT 1 Bayesian optmization

HPModel1= fitctree(Xtr,ytr,'OptimizeHyperparameters','auto')
%% 


%% yn and yr DT for orginall

%% %% This is best train model for DT optimized

OptimisedDT = fitctree(Xtr,ytr,'MinLeafSize',7,'MaxNumSplits',4)
%% predict DT optimized
DTo=      predict(  OptimisedDT,Xte);

%%  DT optimized confusion matrix
DTocm = confusionmat(yte,DTo); % DT optimized confusion matrix
figure(2)
movegui('west')
confusionchart(DTocm);
title('DT optimized Confusion Matrix')




%% %% 2 RF Bayesian optmization

t = templateTree('MaxNumSplits',4,'MinLeafSize', 7);
BOModel1 = fitcensemble(Xtr,ytr,'OptimizeHyperparameters','auto','Learnersc',t);

%% %% %% This is best train model for RF optimized

OptimisedRF = fitcensemble(Xtr,ytr,'MinLeafSize',60,'MaxNumSplits',60) 

%% %%  DT optimized confusion matrix
RFocm = confusionmat(yte,RFo); % RF optimized confusion matrix
figure(2)
movegui('west')
confusionchart(RFocm);
title('RF optimized Confusion Matrix')


%% %% predict RF optimized
RFo=      predict(  OptimisedRF,Xte);




