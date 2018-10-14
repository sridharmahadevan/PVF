function vVals = FastSimpleNNInterp( cPoints, cNewPoints, cF, cNNInfo )

%
% function vVals = FastSimpleNNInterp( cPoints, cNewPoints, cF, cNNInfo )
%
% Simple nearest neighbor interpolation
%
%   cPoints         : M by N matrix of M points in R^N on which the graph and eigenvectors were originally constructed
%   cNewPoints      : M' by N matrix M' points in R^N which the eigenvectors should be extended to.
%   cF              : D by M matrix of D functions on the original M points
%   cNNInfo         : structure containing 
%                       Atria   : structure created by pre-processing nearest neighbors
%                       Delta   : scaling for computing the exponential weights
%                       kNN     : number of nearest neighbors
%
% OUT:
%   vVals           : D by M' matrix with the values of each of the D functions cF evaluated at each of the new M' new cNewPoints.
%
% EXAMPLE:
%   cX=[cos(linspace(0,2*pi,2000));sin(linspace(0,2*pi,2000))]';
%   cX2=cX+0.1*randn(size(cX));
%   [vGraph, vEigenVecs, vEigenVals, vOpts, vLaplacian, vNNInfo, vDiffusion] = FastLaplacianDetEigs( cX, struct('Type','nn','kNN',20,'Delta',0.2,'NormalizationType','graph','MaxEigenVals',100 ) );
%   vVals = FastSimpleNNInterp( cX, cX2, vEigenVecs, struct('Atria',vNNInfo.Atria,'Delta',vOpts.Delta,'kNN',vOpts.kNN) );
%   figure;scatter(cX(:,1),cX(:,2),5,vEigenVecs(40,:),'filled');hold on;scatter(cX2(:,1),cX2(:,2),10,vVals(40,:));
%
% SC:
%   MM      :10/19/2005
%   MM      :11/26/2005 : sped up
%

% Need to interpolate the operator at the new point
% First compute the distance matrix relative to this point.
[lD lM] = size(cF);
lP = size(cNewPoints,1);
vVals = zeros([lD lP]);
% Interpolate to the new points.
[lNNIdxs,lNNDist] = nn_search(cPoints, cNNInfo.Atria, cNewPoints, cNNInfo.kNN);
if cNNInfo.Delta == Inf,
    lNNDist = ones(size(lNNDist));
else
    lNNDist = exp(-(lNNDist/cNNInfo.Delta).^2);
end;
lDegrees = sum(lNNDist,2)';

% SRIDHAR: added this patch to prevent divide by 0 errors! 
if ~isempty(find(lDegrees==0))  % something wrong!
    indices = find(lDegrees==0);
    lDegrees(indices)=1;  % SRIDHAR: 1/21/06: set equal to 1 to prevent divide by 0 error!
    %     disp(cNewPoints);
    %     disp(indices);  pause;
    %     figure; plot(lDegrees);  drawnow; pause;
end;


% Loop through the new points...
%lF = reshape(cF(:,lNNIdxs),[size(cF,1),size(lNNIdxs,1),size(lNNIdxs,2)]);
for lj = 1:lD
    lF = cF(lj,:);
    vVals(lj,:) = sum(lF(lNNIdxs).*lNNDist,2)'./lDegrees;
    %vVals(lj,:) = sum(squeeze(lF(lj,:,:)).*lNNDist,2)./lDegrees';
end;

return;
