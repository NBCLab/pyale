% Create kernels test mat file.
% Must have SPM in path.
Template = spm_vol('./MaskenEtc/Grey10.nii');
uncertainTemplates = (5.7/(2*sqrt(2/pi)) * sqrt(8*log(2)));   % Assuming 5.7 mm ED between templates
sampleSizes = 1:50;

kernels = zeros(31, 31, 31, length(sampleSizes));
for n = sampleSizes
    xleer = zeros(31, 31, 31);
    xleer(16, 16, 16) = 1;
    SomeStruct = struct('dim', [31, 31, 31], 'mat', Template.mat);
    
    uncertainSubjects = (11.6/(2*sqrt(2/pi)) * sqrt(8*log(2))) / sqrt(n);   % Assuming 11.6 mm ED between matching points
    smoothing = sqrt(uncertainSubjects.^2 + uncertainTemplates.^2);
    
    kernels(:, :, :, n) = single(my_MemSmooth64bit(xleer, smoothing, SomeStruct, zeros(31)));
end

save('kernels.mat', 'kernels');
