function hands_on_CV
    
    clear all;
    clc;
    close all;
    
    load Auto.mat
    Auto = rmmissing(Auto);
    % 
    % %% Estrategia del conjunto de validación
    % 
    % %timer
    % tic
    % 
    % rng(20030210)
    % hpartition = cvpartition(size(Auto, 1),"HoldOut",0.5);
    % 
    % % Para ver que elementos van a que grupo
    % pos_train = hpartition.training;
    % pos_test = hpartition.test;
    % 
    % var_sel = 4; % Horsepower
    % var_result = 1; % mpg
    % Xtrain = Auto{pos_train, var_sel};
    % Xtest = Auto{pos_train, var_sel};
    % 
    % Ytrain = Auto{pos_train, var_result};
    % Ytest = Auto{pos_train, var_result};
    % 
    % % Lineal solo
    % mdl_01 = fitlm(Xtrain, Ytrain);
    % 
    % Ypred = predict(mdl_01, Xtest);
    % MSE_01 = mean((Ytest-Ypred).^2);
    % 
    % % Cuadrático
    % mdl_02 = fitlm([Xtrain, Xtrain.^2], Ytrain);
    % 
    % Ypred = predict(mdl_02, [Xtest, Xtest.^2]);
    % MSE_02 = mean((Ytest-Ypred).^2);
    % 
    % % Cúbica
    % mdl_03 = fitlm([Xtrain, Xtrain.^2, Xtrain.^3], Ytrain);
    % 
    % Ypred = predict(mdl_03, [Xtest, Xtest.^2, Xtest.^3]);
    % MSE_03 = mean((Ytest-Ypred).^2);
    % 
    % fprintf("\n\nMSE lineala: %f\nMSE kuadratikoa: %f\nMSE Kubikoa: %f\n", MSE_01, MSE_02, MSE_03);
    % toc
    % 
    % %% LOOCV
    % tic
    % c = cvpartition(size(Auto, 1), 'LeaveOut');
    % X = Auto.horsepower;
    % Y = Auto.mpg;
    % 
    % fprintf("Leave-one-out cross-validation\n")
    % for cc=2:4
    %     X = [X, Auto.horsepower.^cc];
    %     CV_MSE(cc) = crossval('mse', X, Y, 'Predfun', @RLIN, 'partition',c);
    %     fprintf("MSE %d ordeneko modeloarentzat: %f\n", cc, CV_MSE(cc));
    % 
    % end
    % toc
    % 
    % %% K-FOLD CV
    % tic
    % K = 10;
    % c = cvpartition(size(Auto, 1), 'KFold', K);
    % X = Auto.horsepower;
    % Y = Auto.mpg;
    % 
    % fprintf("K-Fold cross-validation\n")
    % for cc=2:4
    %     X = [X, Auto.horsepower.^cc];
    %     CV_MSE(cc) = crossval('mse', X, Y, 'Predfun', @RLIN, 'partition',c);
    %     fprintf("MSE %d ordeneko modeloarentzat: %f\n", cc, CV_MSE(cc));
    % 
    % end
    %
    % toc

    %% Bootstrap
    load("Portfolio.mat")
    rng(20030210)
    d = 1000;
    bootStat = bootstrp(d, @compute_statistic, Portfolio{:,:});
    fprintf("\n Media (SD) de alpha es %.3f (%.3f)\n", mean(bootStat), std(bootStat));

    %% Bootstrap Again
    d = 1000;
    X = Auto.horsepower;
    Y = Auto.mpg;

    fprintf("\nLinear model!\n")

    % Linear model
    rng(20030210)
    bootStat = bootstrp(d, @adjust_RLIN, [Y, X]);
    % We ordered them [Y, X] for funsies

    fprintf("\nMedia (SD) de B0 es %.3f (%.3f)\n", mean(bootStat(:,1)), std(bootStat(:,1)));   
    fprintf("Media (SD) de B1 es %.3f (%.3f)\n", mean(bootStat(:,2)), std(bootStat(:,2)));

    mdl = fitlm(X, Y);
    fprintf("\nA través de fitlm la estimación (SE) de B0 es %.3f (%.3f)\n", mdl.Coefficients.Estimate(1), mdl.Coefficients.SE(1));
    fprintf("A través de fitlm la estimación (SE) de B1 es %.3f (%.3f)\n", mdl.Coefficients.Estimate(2), mdl.Coefficients.SE(2));

    fprintf("\nQuadratic model!\n")

    % Quadratic model
    rng(20030210)
    bootStat = bootstrp(d, @adjust_RLIN, [Y, X, X.^2]);
    % We ordered them [Y, X] for funsies

    fprintf("\nMedia (SD) de B0 es %.3f (%.3f)\n", mean(bootStat(:,1)), std(bootStat(:,1)));   
    fprintf("Media (SD) de B1 es %.3f (%.3f)\n", mean(bootStat(:,2)), std(bootStat(:,2)));
    fprintf("Media (SD) de B2 es %.3f (%.3f)\n", mean(bootStat(:,3)), std(bootStat(:,3)));

    mdl = fitlm([X, X.^2], Y);
    fprintf("\nA través de fitlm la estimación (SE) de B0 es %.3f (%.3f)\n", mdl.Coefficients.Estimate(1), mdl.Coefficients.SE(1));
    fprintf("A través de fitlm la estimación (SE) de B1 es %.3f (%.3f)\n", mdl.Coefficients.Estimate(2), mdl.Coefficients.SE(2));
    fprintf("A través de fitlm la estimación (SE) de B2 es %.3f (%.3f)\n", mdl.Coefficients.Estimate(3), mdl.Coefficients.SE(3));

end

%% User defined functions
function yfit = RLIN(Xtrain, Ytrain, Xtest)
    
    mdl = fitlm(Xtrain, Ytrain);
    yfit = predict(mdl, Xtest);
    
end

function alpha = compute_statistic(samples)
    X = samples(:,1);
    Y = samples(:,2);
    CovM = cov(X, Y);
    % var(Y) = CovM(2,2)!!!
    alpha = (CovM(2,2)-CovM(2,1))/(CovM(1,1) + CovM(2,2) - 2*CovM(1,2));
end

function coef = adjust_RLIN(samples)
    
    % We ordered them [Y, X] for funsies
    mdl = fitlm(samples(:,2:end), samples(:,1));
    coef = mdl.Coefficients.Estimate;

end