function [precision, recall] = PRCurve(smapImg, gtImg)
% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014

if ~islogical(gtImg)
    gtImg = gtImg(:,:,1) > 128;
end
if any(size(smapImg) ~= size(gtImg))
    error('saliency map and ground truth mask have different size');
end

gtPxlNum = sum(gtImg(:));
if 0 == gtPxlNum
    error('no foreground region is labeled');
end

targetHist = histc(smapImg(gtImg), 0:255);
nontargetHist = histc(smapImg(~gtImg), 0:255);

targetHist = flipud(targetHist);
nontargetHist = flipud(nontargetHist);

targetHist = cumsum( targetHist );
nontargetHist = cumsum( nontargetHist );

precision = targetHist ./ (targetHist + nontargetHist + eps);
if any(isnan(precision))
    warning('there exists NAN in precision, this is because  your saliency map do not range from 0 to 255\n');
end
recall = targetHist / gtPxlNum;
