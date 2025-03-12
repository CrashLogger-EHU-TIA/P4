function hands_on_CV
    
    clear all;
    clc;
    close all;
    
    load Auto.mat
    Auto = rmmissing(Auto);
    
    %% Estrategia del conjunto de validación
    
    %timer
    tic
    
    rng(20030210)
    hpartition = cvpartition(size(Auto, 1),"HoldOut",0.5);
    
    % Para ver que elementos van a que grupo
    pos_train = hpartition.training;
    pos_test = hpartition.test;
    
    var_sel = 4; % Horsepower
    var_result = 1; % mpg
    Xtrain = Auto{pos_train, var_sel};
    Xtest = Auto{pos_train, var_sel};
    
    Ytrain = Auto{pos_train, var_result};
    Ytest = Auto{pos_train, var_result};
    
    % Lineal solo
    mdl_01 = fitlm(Xtrain, Ytrain);
    
    Ypred = predict(mdl_01, Xtest);
    MSE_01 = mean((Ytest-Ypred).^2);
    
    % Cuadrático
    mdl_02 = fitlm([Xtrain, Xtrain.^2], Ytrain);
    
    Ypred = predict(mdl_02, [Xtest, Xtest.^2]);
    MSE_02 = mean((Ytest-Ypred).^2);
    
    % Cúbica
    mdl_03 = fitlm([Xtrain, Xtrain.^2, Xtrain.^3], Ytrain);
    
    Ypred = predict(mdl_03, [Xtest, Xtest.^2, Xtest.^3]);
    MSE_03 = mean((Ytest-Ypred).^2);
    
    fprintf("\n\nMSE lineala: %f\nMSE kuadratikoa: %f\nMSE Kubikoa: %f\n", MSE_01, MSE_02, MSE_03);
    toc
    
    %% LOOCV
    tic
    c = cvpartition(size(Auto, 1), 'LeaveOut');
    X = Auto.horsepower;
    Y = Auto.mpg;
    
    for cc=2:4
        X = [X, Auto.horsepower.^cc];
        CV_MSE(cc) = crossval('mse', X, Y, 'Predfun', @RLIN, 'partition',c);
        fprintf("MSE %d ordeneko modeloarentzat: %f\n", cc, CV_MSE(cc));

    end
    
    toc
end

function yfit = RLIN(Xtrain, Ytrain, Xtest)
    
    mdl = fitlm(Xtrain, Ytrain);
    yfit = predict(mdl, Xtest);
    
end