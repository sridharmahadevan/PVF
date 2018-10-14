function [vIdxs,vInvIdxs,vD,vS,vV] = FastRoughSetDownsample( cX, cDelta, cSigma, cDownSample, cDoSVD, cSVDDelta )

%
% function [vIdxs,vD] = FastRoughSetDownsample( cX, cDelta, cSigma, cDoSVD )
%
% IN:
%   cX      : M by N matrix of M points in N dimensions
%   cDelta  : find a maximal cDelta net
%   cSigma  : exponential weights for each chosen point: e^(-(dist/cSigma)^2)
%   [cDoSVD]: compute local svd. Default: false.
%   [cSVDDelta] : compute SVD on points in a delta ball around each point. Default: cDelta.
%
% OUT:
%   vIdxs   : cX(vIdxs,:) is the maximal cDelta net
%   vInvIdxs: vInvIdxs is an M vector, and cX(vInvIdxs,:) is set-wise equal to cX(vIdxs,:)
%   vD      : vD(k) is the mass to assign to the k-th selected point cX(vIdxs(k),:)
%
% USES:
%   nn_prepare, range_search        : from OpenTSTool, for fast range_searches
%
% EXAMPLE:
%   lParam=linspace(0,2*pi,100000);cX=[sin(lParam)',cos(lParam)']+0.01*randn(100000,2);
%   [vIdxs,vInvIdxs] = FastRoughSetDownsample( cX, 0.2 );figure;plot(cX(:,1),cX(:,2),'bo');hold on;plot(cX(vIdxs,1),cX(vIdxs,2),'go');hold on;plot(cX(vIdxs(10),1),cX(vIdxs(10),2),'ro');plot(cX(find(vInvIdxs==vIdxs(10)),1),cX(find(vInvIdxs==vIdxs(10)),2),'k.');
%
%   % Note: this runs in 0.83 sec on my laptop, P5 3Ghz, 1Gb RAM.
%   % One million points runs in 8.4 seconds...seems really linear in low dimensions
%
% SC:
%   MM  : 5/13/06       : small optimization
%
% Copyright 2005
% 
% Mauro Maggioni
% mauro@math.duke.edu
%

if nargin < 3,
    cSigma = cDelta;
end;
if nargin < 4,
    cDownSample = true;
end;
if nargin < 5,
    cDoSVD = false;
end;
if nargin < 6,
    cSVDDelta = cDelta;
end;

[lNumberOfPoints lDim] = size(cX);

% Allocate memory
vIdxs(lNumberOfPoints) = int32(0);
vInvIdxs(lNumberOfPoints) = int32(0);
vD(lNumberOfPoints) = double(0);

if cDoSVD,
    vS(lNumberOfPoints,lDim) = double(0);
end;

% Prepare the set for fast range_searches
lAtria = nn_prepare(cX,'euclidian',32);

lNotChosenIdxs = 1:lNumberOfPoints;
lChosenIdx = 1;

while (length(lNotChosenIdxs)>0),
    % Pick a point in a good position
    lPoint = cX(lNotChosenIdxs(1),:);
    % Find its cDelta-neighbors
    [lNNCount, lNN] = range_search(cX,lAtria,lNotChosenIdxs(1),cDelta,-1);        
    % Indices of the points...
    lCloseIdxs = lNN{1,1};
    % Save the index of point selected as index into rows of cX
    vIdxs(lChosenIdx) = lNotChosenIdxs(1);
    vInvIdxs(lCloseIdxs) = lChosenIdx;
    if nargout>1,
        % ...and their distances, exponentially weighted
        lCloseW = exp(-(lNN{1,2}/cSigma).^2);
        % Assign the degree to the vertex.
        vD(lChosenIdx) = sum(lCloseW);
    end;
    % ..compute the SVD if requested
    if cDoSVD,
        [lNNCount, lNN] = range_search(cX,lAtria,lNotChosenIdxs(1),cSVDDelta,-1);
        lCloseIdxs = lNN{1,1};
        if lCloseIdxs > 1,
            % Throw away the close points which are more than 1.5 times far than the closest point
            lReallyCloseIdxs = lNN{1,1}(find(lNN{1,2}<=median(lNN{1,2})));            
            [lU lS lV] = svd(cX(lReallyCloseIdxs,:)-repmat(lPoint,[length(lReallyCloseIdxs),1]),0);
            lS=diag(lS);
            if( length(lS) == size(vS,2)) & (size(lS,2)==1),
                vS(lChosenIdx,:) = lS;
            else
                vS(lChosenIdx,:) = 0;
            end;
            vV(lChosenIdx,:,:) = lV;
        else
            vS(lChosenIdx,:) = 0;
            %vV(lChosenIdx,:,:) = 0;
        end;
    end;    
    % The point selected and its cDelta-neighbors shouldn't be selected again:
    % take them off the list of available good points   
    if cDownSample,
        lNotChosenIdxs = lNotChosenIdxs(~(ismember(lNotChosenIdxs,union(lCloseIdxs,lNotChosenIdxs(1)))));    % Faster version (though unreadable!) instead of lNotChosenIdxs = setdiff(lNotChosenIdxs,lCloseIdxs);
    else
        lNotChosenIdxs(1)=[];
    end;
    % Go to the next point
    lChosenIdx = lChosenIdx+1;
    %figure(1);plot(length(vIdxs),length(lNotChosenIdxs),'o');hold on;drawnow;
end;

if nargout>1,
    vD(lChosenIdx:length(vIdxs))=[];
end;
vIdxs(lChosenIdx:length(vIdxs))=[];
if cDoSVD,
    vS(lChosenIdx:length(vIdxs),:) = [];
end;

return;




























% This is the first working version
if nargin < 3,
    cSigma = cDelta;
end;

[lNumberOfPoints lDim] = size(cX);

lXNotChosen = cX;
lNotChosenIdxs = 1:lNumberOfPoints;
lChosenIdx = 1;
cDelta=cDelta^2;
while (length(lNotChosenIdxs)>0),
    lPoint = cX(lNotChosenIdxs(1),:);
    lDist = [];
    for lj = length(lNotChosenIdxs):-1:1,
        lDist(lj) = sum( (lPoint-cX(lNotChosenIdxs(lj),:)).^2 );
    end;
    lCloseIdxs = find(lDist<cDelta);
    lCloseW = exp(-(lDist(lCloseIdxs)/cSigma).^2);
    vD(lChosenIdx) = sum(lCloseW);
    vIdxs(lChosenIdx) = lNotChosenIdxs(1);
    lChosenIdx = lChosenIdx+1;

    lNotChosenIdxs(lCloseIdxs) = [];
end;



% End of the first working version









































% SEcond working version
if nargin < 3,
    cSigma = cDelta;
end;

[lNumberOfPoints lDim] = size(cX);

vIdxs(lNumberOfPoints) = int32(0);
vD(lNumberOfPoints) = double(0);

lAtria = nn_prepare(cX);

lXNotChosen = cX;
lNotChosenIdxs = 1:lNumberOfPoints;
lChosenIdx = 1;

lX = cX;
lIndirect = 1:size(cX,1);

while (length(lNotChosenIdxs)>0),
    lPoint = lX(lNotChosenIdxs(1),:);
    [lNNCount, lNN] = range_search(lX,lAtria,lNotChosenIdxs(1),cDelta,-1);        
    lCloseIdxs = lNN{1,1};
    lCloseW = exp(-(lNN{1,2}/cSigma).^2);
    vD(lChosenIdx) = sum(lCloseW);
    vIdxs(lChosenIdx) = lIndirect(lNotChosenIdxs(1));
    lChosenIdx = lChosenIdx+1;
        
    lTempIdxs = ~(ismember(lNotChosenIdxs,lCloseIdxs)); %lNotChosenIdxs = setdiff(lNotChosenIdxs,lCloseIdxs);
    lNotChosenIdxs = lNotChosenIdxs(lTempIdxs);    
    
%    if (length(lNotChosenIdxs)<0.1*size(lX,1)) & (length(lNotChosenIdxs)>500),                 % This affects the weights...
%        lX = lX(lNotChosenIdxs,:);
%        lAtria = nn_prepare(lX);
%        lIndirect = lIndirect(lNotChosenIdxs);
%        lNotChosenIdxs = 1:length(lIndirect);
%    end;
end;

vD(lChosenIdx:length(vIdxs))=[];
vIdxs(lChosenIdx:length(vIdxs))=[];

return;






























% Second not working cversion
if nargin < 3,
    cSigma = cDelta;
end;

[lNumberOfPoints lDim] = size(cX);

lBinEdges = cDelta:cDelta:1000*cDelta;

lXNotChosen = cX;
lNotChosenIdxs = 1:lNumberOfPoints;
lChosenIdx = 1;
figure;
while (length(lNotChosenIdxs)>0),
    cla;scatter(cX(:,1),cX(:,2),50,'b');hold on;
    % A new point that wasn't chosen is added.
    lPoint = cX(lNotChosenIdxs(1),:);        
    % Compute the distances between this points and all the points in the set.
    lDist = [];
    for lj = length(lNotChosenIdxs):-1:1,
        lDist(lj) = sum( (lPoint-cX(lNotChosenIdxs(lj),:)).^2 );
    end;
    lDist = sqrt(lDist);
    [lSortedDist,lSortedDistIdxs]=sort(lDist);
    lBreakPoints = BinSortedArray(lSortedDist,lBinEdges);
    lBreakPoints = lBreakPoints(find(lBreakPoints~=0));
    % Pick a point in each other other bin
    lNotToBeChosenIdxs = [];
    for lk = 1:3:length(lBreakPoints),
        if lBreakPoints(lk)==0,continue;end;
        % Choose a point in the bin        
        vIdxs(lChosenIdx) = lNotChosenIdxs(lSortedDistIdxs(lBreakPoints(lk)));
        % Find all the closest points: they're either in the previous bin, in the same bin, or the next bin
        if lk>1, lBottomIdx = lBreakPoints(lk-1); else lBottomIdx = 1; end;
        if lk+1<length(lBreakPoints), lTopIdx=lBreakPoints(lk+1); else lTopIdx = length(lSortedDist); end;
        lCloseIdxs = lBottomIdx:lTopIdx;
        % Compute the distances from all the close candidates
        lCloseDist = [];
        for li = length(lCloseIdxs):-1:1;
            lCloseDist(li) = sqrt(sum((cX(lNotChosenIdxs(lSortedDistIdxs(lCloseIdxs(li))),:)-cX(vIdxs(lChosenIdx),:)).^2));%            figure(2);scatter(cX(lNotChosenIdxs(lSortedDistIdxs(lCloseIdxs(li))),1),cX(lNotChosenIdxs(lSortedDistIdxs(lCloseIdxs(li))),2),200,'k');hold on;scatter(cX(vIdxs(lChosenIdx),1),cX(vIdxs(lChosenIdx),2),200,'b');pause            
        end;
        % Find the cDelta-close points among the candidates
        lDeltaCloseIdxs =lCloseIdxs(find(lCloseDist<cDelta));
        % Compute the weight of the new point
        vD(lChosenIdx) = sum(exp(-(lSortedDist(lDeltaCloseIdxs)/cSigma).^2));
        % Erase from the list of interesting points the ones close to the selected point
        lNotToBeChosenIdxs = union(lNotToBeChosenIdxs,lSortedDistIdxs(lDeltaCloseIdxs));
        %figure(2);cla;scatter(cX(lNotChosenIdxs(lSortedDistIdxs(lCloseIdxs)),1),cX(lNotChosenIdxs(lSortedDistIdxs(lCloseIdxs)),2),50,'b');hold on;
        %scatter(cX(lNotChosenIdxs(lSortedDistIdxs(lDeltaCloseIdxs)),1),cX(lNotChosenIdxs(lSortedDistIdxs(lDeltaCloseIdxs)),2),50,'r');hold on;
        %scatter(cX(vIdxs(lChosenIdx),1),cX(vIdxs(lChosenIdx),2),250,'g');        pause;
        lChosenIdx = lChosenIdx + 1;
    end;       
    figure(1);
    lNotChosenIdxs(lNotToBeChosenIdxs) = [];    
    scatter(cX(lNotChosenIdxs,1),cX(lNotChosenIdxs,2),50,'r','o');
    scatter(cX(vIdxs,1),cX(vIdxs,2),50,'g','filled');
    fprintf('Chosen %d, not chosen: %d\n',length(lChosenIdx),length(lNotChosenIdxs));    pause;
end;