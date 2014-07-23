% data quality check NYU tools

analysis_path = getenv('ANALYSIS_HOME');

addpath(genpath(strcat(analysis_path, 'Tools/other_scripts/fMRI_quality/dataQuality-1.5/')))
addpath(strcat(analysis_path, 'Tools/other_scripts/fMRI_quality/NIfTImatlab/matlab'))

cd('---FUNC_FILE_DIR---')

datafName = '---FUNC_FILE---';
gunzip(datafName)

roifName = 'rois.txt';
%% run quality check:
roiCorners(datafName(1:end-3), roifName)
dataQReport(datafName(1:end-3), roifName)

%% delete stuff:
delete(datafName(1:end-3))

exit