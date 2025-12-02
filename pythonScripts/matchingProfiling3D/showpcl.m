figure(1)
clf
ptCloud = pcread("resultingTransformationGT.ply");


pcshow(ptCloud)



figure(2)
clf
ptCloud = pcread("resultingTransformationHighestEstimation.ply");


pcshow(ptCloud)



figure(3)
clf
ptCloud = pcread("resultingTransformationBestEstimation.ply");


pcshow(ptCloud)