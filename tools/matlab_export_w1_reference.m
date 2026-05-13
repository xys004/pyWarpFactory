% Export W1_Warp_Shell reference arrays from MATLAB WarpFactory.
%
% Run from the repository root in MATLAB:
%   run('tools/matlab_export_w1_reference.m')
%
% This writes tools/w1_reference.mat with the same recipe used by
% Examples/4 Warp Shell/W1_Warp_Shell.mlx. Convert to .npz or compare from
% Python after loading the MAT file.

spaceScale = 5;
timeScale = 1;
centered = 1;
cartoonThickness = 5;

R1 = 10;
Rbuff = 0;
R2 = 20;
factor = 1/3;
m = R2/(2*G)*c^2*factor;
vWarp = 0.02;
sigma = 0;
doWarp = 1;
smoothFactor = 4000;

if centered == 1
    gridSize = ceil([1,2*(R2+10)*spaceScale,2*(R2+10)*spaceScale,cartoonThickness]);
else
    gridSize = ceil([1,(R2+10)*spaceScale,(R2+10)*spaceScale,cartoonThickness]);
end

gridScaling = [1/(timeScale*spaceScale*((vWarp)*c+1)),1/spaceScale,1/spaceScale,1/spaceScale];
gridScaling(1) = 1/(1000*c);

if centered == 1
    worldCenter = [(cartoonThickness+1)/2,(2*(R2+10)*spaceScale+1)/2,(2*(R2+10)*spaceScale+1)/2,(cartoonThickness+1)/2].*gridScaling;
else
    worldCenter = [(cartoonThickness+1)/2,5,5,(cartoonThickness+1)/2].*gridScaling;
end

Metric = metricGet_WarpShellComoving(gridSize,worldCenter,m,R1,R2,Rbuff,sigma,smoothFactor,vWarp,doWarp,gridScaling);
ConstantWarp = evalMetric(Metric,1,1);

MetricTensor = zeros([4, 4, gridSize]);
EnergyEulerianTensor = zeros([4, 4, gridSize]);
for mu = 1:4
    for nu = 1:4
        MetricTensor(mu, nu, :, :, :, :) = Metric.tensor{mu, nu};
        EnergyEulerianTensor(mu, nu, :, :, :, :) = ConstantWarp.energyTensorEulerian.tensor{mu, nu};
    end
end

r_sample = Metric.params.rVec;
rho = Metric.params.rho;
rho_smooth = Metric.params.rhosmooth;
P = Metric.params.P;
P_smooth = Metric.params.Psmooth;
M = Metric.params.M;
A = Metric.params.A;
B = Metric.params.B;
NullMap = ConstantWarp.null;
WeakMap = ConstantWarp.weak;
StrongMap = ConstantWarp.strong;
DominantMap = ConstantWarp.dominant;

outputPath = fullfile('tools', 'w1_reference.mat');
save(outputPath, ...
    'R1', 'R2', 'Rbuff', 'factor', 'm', 'vWarp', 'sigma', 'doWarp', ...
    'smoothFactor', 'spaceScale', 'timeScale', 'gridSize', 'gridScaling', ...
    'worldCenter', 'MetricTensor', 'EnergyEulerianTensor', ...
    'r_sample', 'rho', 'rho_smooth', 'P', 'P_smooth', 'M', 'A', 'B', ...
    'NullMap', 'WeakMap', 'StrongMap', 'DominantMap', '-v7.3');

fprintf('Saved W1 reference arrays to %s\n', outputPath);
