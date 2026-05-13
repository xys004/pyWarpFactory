% Export static and shifted W1 component-audit arrays from MATLAB WarpFactory.
%
% Run from the repository root in MATLAB:
%   run('tools/matlab_export_w1_component_audit.m')
%
% This writes tools/w1_component_audit_reference.mat. Convert with
% tools/convert_w1_mat_to_npz.py or load directly from Python.

spaceScale = 5;
timeScale = 1;
centered = 1;
cartoonThickness = 5;

R1 = 10;
Rbuff = 0;
R2 = 20;
factor = 1/3;
m = R2/(2*G)*c^2*factor;
sigma = 0;
smoothFactor = 4000;
numAngularVec = 4;
numTimeVec = 10;

if centered == 1
    gridSize = ceil([1,2*(R2+10)*spaceScale,2*(R2+10)*spaceScale,cartoonThickness]);
else
    gridSize = ceil([1,(R2+10)*spaceScale,(R2+10)*spaceScale,cartoonThickness]);
end

gridScaling = [1/(1000*c),1/spaceScale,1/spaceScale,1/spaceScale];

if centered == 1
    worldCenter = [(cartoonThickness+1)/2,(2*(R2+10)*spaceScale+1)/2,(2*(R2+10)*spaceScale+1)/2,(cartoonThickness+1)/2].*gridScaling;
else
    worldCenter = [(cartoonThickness+1)/2,5,5,(cartoonThickness+1)/2].*gridScaling;
end

cases = ["static", "shifted"];
velocities = [0, 0.02];
doWarps = [0, 1];

for caseIdx = 1:length(cases)
    caseName = cases(caseIdx);
    vWarp = velocities(caseIdx);
    doWarp = doWarps(caseIdx);

    Metric = metricGet_WarpShellComoving(gridSize,worldCenter,m,R1,R2,Rbuff,sigma,smoothFactor,vWarp,doWarp,gridScaling);
    MetricInverse = c4Inv(Metric.tensor);
    RicciTensor = ricciT(MetricInverse, Metric.tensor, gridScaling);
    RicciScalar = ricciS(RicciTensor, MetricInverse);
    EinsteinTensor = einT(RicciTensor, RicciScalar, Metric.tensor);
    ConstantWarp = evalMetric(Metric,1,1);

    metricTensor = zeros([4, 4, gridSize]);
    ricciTensor = zeros([4, 4, gridSize]);
    einsteinTensor = zeros([4, 4, gridSize]);
    energyTensor = zeros([4, 4, gridSize]);
    energyEulerianTensor = zeros([4, 4, gridSize]);
    for mu = 1:4
        for nu = 1:4
            metricTensor(mu, nu, :, :, :, :) = Metric.tensor{mu, nu};
            ricciTensor(mu, nu, :, :, :, :) = RicciTensor{mu, nu};
            einsteinTensor(mu, nu, :, :, :, :) = EinsteinTensor{mu, nu};
            energyTensor(mu, nu, :, :, :, :) = ConstantWarp.energyTensor.tensor{mu, nu};
            energyEulerianTensor(mu, nu, :, :, :, :) = ConstantWarp.energyTensorEulerian.tensor{mu, nu};
        end
    end

    nullMap = getEnergyConditions(ConstantWarp.energyTensor, Metric, "Null", numAngularVec, numTimeVec, 0, 0);
    weakMap = getEnergyConditions(ConstantWarp.energyTensor, Metric, "Weak", numAngularVec, numTimeVec, 0, 0);
    strongMap = getEnergyConditions(ConstantWarp.energyTensor, Metric, "Strong", numAngularVec, numTimeVec, 0, 0);
    dominantMap = getEnergyConditions(ConstantWarp.energyTensor, Metric, "Dominant", numAngularVec, numTimeVec, 0, 0);

    assignin('base', char(caseName + "_MetricTensor"), metricTensor);
    assignin('base', char(caseName + "_RicciTensor"), ricciTensor);
    assignin('base', char(caseName + "_RicciScalar"), RicciScalar);
    assignin('base', char(caseName + "_EinsteinTensor"), einsteinTensor);
    assignin('base', char(caseName + "_EnergyTensor"), energyTensor);
    assignin('base', char(caseName + "_EnergyEulerianTensor"), energyEulerianTensor);
    assignin('base', char(caseName + "_NullMap"), nullMap);
    assignin('base', char(caseName + "_WeakMap"), weakMap);
    assignin('base', char(caseName + "_StrongMap"), strongMap);
    assignin('base', char(caseName + "_DominantMap"), dominantMap);
end

outputPath = fullfile('tools', 'w1_component_audit_reference.mat');
save(outputPath, ...
    'R1', 'R2', 'Rbuff', 'factor', 'm', 'sigma', 'smoothFactor', ...
    'spaceScale', 'timeScale', 'gridSize', 'gridScaling', 'worldCenter', ...
    'numAngularVec', 'numTimeVec', ...
    'static_MetricTensor', 'static_RicciTensor', 'static_RicciScalar', ...
    'static_EinsteinTensor', 'static_EnergyTensor', 'static_EnergyEulerianTensor', ...
    'static_NullMap', 'static_WeakMap', 'static_StrongMap', 'static_DominantMap', ...
    'shifted_MetricTensor', 'shifted_RicciTensor', 'shifted_RicciScalar', ...
    'shifted_EinsteinTensor', 'shifted_EnergyTensor', 'shifted_EnergyEulerianTensor', ...
    'shifted_NullMap', 'shifted_WeakMap', 'shifted_StrongMap', 'shifted_DominantMap', ...
    '-v7.3');

fprintf('Saved W1 component audit reference arrays to %s\n', outputPath);
