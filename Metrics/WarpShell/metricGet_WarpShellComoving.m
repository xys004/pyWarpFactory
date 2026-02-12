function [Metric] = metricGet_WarpShellComoving(gridSize,worldCenter,m,R1,R2,Rbuff,sigma,smoothFactor,vWarp,doWarp,gridScaling)


%% METRICGET_WARPSHELLCOMOVING: Builds the Warp Shell metric in a comoving frame
% https://iopscience.iop.org/article/10.1088/1361-6382/ad26aa
%
%   INPUTS: 
%   gridSize - 1x4 array. world size in [t, x, y, z], double type.
%   
%   worldCenter - 1x4 array. world center location in [t, x, y, z], double type.
%
%   m - total mass of the warp shell
%
%   R1 - inner radius of the shell
%
%   R2 - outer radius of the shell
%
%   Rbuff - buffer distance between the shell wall and when the shift
%   starts to change
%
%   sigma - sharpness parameter of the shift sigmoid
%
%   smoothfactor - factor by which to smooth the walls of the shell
% 
%   vWarp - speed of the warp drive in factors of c, along the x direction, double type.
% 
%   doWarp - 0 or 1, whether or not to create the warp effect inside the
%   shell
% 
%   gridScale - scaling of the grid in [t, x, y, z]. double type.
%
%   OUTPUTS: 
%   metric - metric struct object. 

%%

% input values
if nargin < 6
    Rbuff = 0;
end
if nargin < 7
    sigma = 0;
end
if nargin < 8
    smoothFactor = 1;
end
if nargin < 9
    vWarp = 0;
end
if nargin < 10
    doWarp = 0;
end
if nargin < 11
    gridScaling = [1,1,1,1];
end

Metric.type = "metric";
Metric.name = "Comoving Warp Shell";
Metric.scaling = gridScaling;
Metric.coords = "cartesian";
Metric.index = "covariant";
Metric.date = date;

% declare radius array
worldSize = sqrt((gridSize(2)*gridScaling(2)-worldCenter(2))^2+(gridSize(3)*gridScaling(3)-worldCenter(3))^2+(gridSize(4)*gridScaling(4)-worldCenter(4))^2);
rSampleRes = 10^5;
rsample = linspace(0,worldSize*1.2,rSampleRes);

% construct rho profile
rho = zeros(1,length(rsample))+m/(4/3*pi*(R2^3-R1^3)).*(rsample>R1 & rsample<R2);
Metric.params.rho = rho;

[~, maxR] = min(diff(rho>0));
maxR = rsample(maxR);

% construct mass profile
M = cumtrapz(rsample, 4*pi.*rho.*rsample.^2);

% construct pressure profile
P = TOVconstDensity(R2,M,rho,rsample);
Metric.params.P = P;

% smooth functions
rho = smooth(smooth(smooth(smooth(rho,1.79*smoothFactor),1.79*smoothFactor),1.79*smoothFactor),1.79*smoothFactor);
rho = rho';
Metric.params.rhosmooth = rho;

P = smooth(smooth(smooth(smooth(P,smoothFactor),smoothFactor),smoothFactor),smoothFactor);
P = P';
Metric.params.Psmooth = P;

% reconstruct mass profile
M = cumtrapz(rsample, 4*pi.*rho.*rsample.^2);
M(M<0) = max(M);


% save varaibles
Metric.params.M = M;
Metric.params.rVec = rsample;


% set shift line vector
shiftRadialVector = compactSigmoid(rsample,R1,R2,sigma,Rbuff);
shiftRadialVector = smooth(smooth(shiftRadialVector,smoothFactor),smoothFactor);

% construct metric using spherical symmetric solution: 
% solve for B
B = (1-2*G.*M./rsample/c^2).^(-1);
B(1) = 1;

% solve for a
a = alphaNumericSolver(M,P,maxR,rsample);

% solve for A from a
A = -exp(2.*a);

% save variables to the metric.params
Metric.params.A = A;
Metric.params.B = B;

% return metric boosted and in cartesian space
Metric.tensor = cell(4);
for mu = 1:4
    for nu = 1:4
        Metric.tensor{mu,nu} = zeros(gridSize);
    end
end
ShiftMatrix = zeros(gridSize);

% set offset value to handle r = 0
epsilon = 0;

for i = 1:gridSize(2)
    for j = 1:gridSize(3)
        for k = 1:gridSize(4)

            x = ((i*gridScaling(2)-worldCenter(2)));
            y = ((j*gridScaling(3)-worldCenter(3)));
            z = ((k*gridScaling(4)-worldCenter(4)));

            %ref Catalog of Spacetimes, Eq. (1.6.2) for coords def.
            r = sqrt(x^2+y^2+z^2)+epsilon;
            theta = atan2(sqrt(x^2+y^2),z);
            phi = atan2(y,x);

            [~, minIdx] = min(abs(rsample-r));
            if rsample(minIdx) > r
                minIdx = minIdx - 1;
            end

            minIdx = minIdx + (r-rsample(minIdx))/(rsample(minIdx+1)-rsample(minIdx));

            g11_sph = legendreRadialInterp(A,minIdx);
            g22_sph = legendreRadialInterp(B,minIdx);

            [g11_cart, g22_cart, g23_cart, g24_cart, g33_cart, g34_cart, g44_cart] = sph2cartDiag(theta,phi,g11_sph,g22_sph);

            Metric.tensor{1,1}(1,i,j,k) = g11_cart;

            Metric.tensor{2,2}(1,i,j,k) = g22_cart;

            Metric.tensor{2,3}(1,i,j,k) = g23_cart;
            Metric.tensor{3,2}(1,i,j,k) = Metric.tensor{2,3}(1,i,j,k);

            Metric.tensor{2,4}(1,i,j,k) = g24_cart;
            Metric.tensor{4,2}(1,i,j,k) = Metric.tensor{2,4}(1,i,j,k);

            Metric.tensor{3,3}(1,i,j,k) = g33_cart;

            Metric.tensor{3,4}(1,i,j,k) = g34_cart;
            Metric.tensor{4,3}(1,i,j,k) = Metric.tensor{3,4}(1,i,j,k);

            Metric.tensor{4,4}(1,i,j,k) = g44_cart; 

            ShiftMatrix(1,i,j,k) = legendreRadialInterp(shiftRadialVector,minIdx);
            
        end
    end
end

% Add warp effect
if doWarp
    Metric.tensor{1,2} = Metric.tensor{1,2}-Metric.tensor{1,2}.*ShiftMatrix - ShiftMatrix*vWarp;
    Metric.tensor{2,1} = Metric.tensor{1,2};
end

end
