function out = dataQReport(datafName,roifName,...
                                  tExclude, defFontSize, defLineWidth, defMarkerSize)

% dataQReport
%
%     out = dataQReport(datafName,roifName)
%
%     out = dataQReport(datafName,roifName,...
%                                  tExclude, defFontSize, defLineWidth, defMarkerSize)
%
% Function to generate an html report with the results of the check for
%   data quality in a fMRI series
%
% arguments:
%   - 'datafName': file containing the data to be analyzed. It supports
%     NIfTI and DICOM formats (if DICOM, 'datafName' could be any of the
%     files of the series).
%   - 'roifName': text file with the coordinates of the corners of the
%     different ROIs to be considered in the statistical analysis.  Use
%     'roiCorners.m' to generate it.
%   - 'tExclude': number of volumes to exclude at the beginning of the run
%     (optional; default=0)
%   - By passing the optional arguments defFontSize, defLineWidth and
%     defMarkerSize you can change the appearance of the figures.
%
% The output structure "out" contains:
%   - snr0
%   - sfnr
%   - sgr
%   - sDate: scan date
%   - coil: coil used for the scan
%   - peaks
%
%
% Dependencies: 
% - For NIfTI images, the matlab-nifti functions are needed;
% - For DICOM images, the image toolbox is needed (both CNS and NYU's
%   licenses have it);
% - 'roiCorners', 'checkDataQuality', 'getNSlices' (they should be in the
%   same package as this function).
%
% PJV: v.1.6: April 2012

% PJV: v.1.6: April 2012
%   Fixed a bug in dataQReport, when calling checkdataquality.
% PJV: v.1.5: April 2012
%   Allows the user to exclude a few time-points at the beginning of the
%   run.
% PJV: v.1.4: March 2009
%   It returns the "out" structure (see above)
% PJV: v.1.3: February 2009
%   If spikes are detected, save the 'peaks' structure containing them in a
%     .mat file in the same folder as the report and the rois.  This file
%     will be an input to 'correct_spikes.m'.
%   Bug fixed: when data path contained a folder, the fName for the html
%     report was wrong.
% PJV: v.1.2: January 2009
%   Force to pass the ROI file name (to make users run roiCorners before
%     generating the report and look at the positioning).
% PJV: v.1.1: December 2008
%   Write the html with fopen and fwrite.
% PJV: v.1.0: December 2008

if (nargin<6)
  defMarkerSize = 10;     % Default marker size (for figures)
  if (nargin<5)
    defLineWidth = 1.2;   % Default line width (for figures)
    if (nargin<4)
      defFontSize = 12;   % Default font size (for labels)
      if (nargin<3)
        tExclude = 0;     % Default number of timepoints to exclude
      end
    end
  end
end

% if no 'roifName' is passed (or blank), return error.
if ( (nargin<2) || isempty(roifName) )
  error('You need to specify a "roifName".');
end
% if 'roifName' doesn't exist, ask user to run 'roiCorners.m' to generate it:
if ~exist(roifName, 'file')
  error('ROI File not found.  Check the filename or run "roiCorners.m"');
end

[pathstr, name, ext] = fileparts(datafName);



%% call 'checkdataquality':

out = checkdataquality(datafName,roifName,...
                       tExclude, defFontSize, defLineWidth, defMarkerSize);
                     

%% save the figures:
% checkdataquality returns the handles

% make 'qualityReport' folder in same directory as datafName:
qRepRoot = ['qReport_' name];
qRepDir = fullfile( pathstr, qRepRoot );
mkdir( qRepDir );
mkdir( fullfile(qRepDir,'figures') )


nFigs = length(out.handles);

print(out.handles(1),'-dpng', fullfile(qRepDir,'figures','h_roi_loc.png'))   % ROI location
print(out.handles(2),'-dpng', fullfile(qRepDir,'figures','h_m_obj.png'))     % mean inten. object
print(out.handles(3),'-dpng', fullfile(qRepDir,'figures','h_m_bkg.png'))     % mean inten. background
print(out.handles(4),'-dpng', fullfile(qRepDir,'figures','h_m_gho.png'))     % mean inten. ghost
print(out.handles(5),'-dpng', fullfile(qRepDir,'figures','h_m_obj_gho.png')) % mean inten. obj linked to ghost
if nFigs>5
  print(out.handles(6),'-dpng', fullfile(qRepDir,'figures','h_spikes.png'))  % background for spikes
  if nFigs>6
    for i=7:nFigs
      print(out.handles(i),'-dpng', ...
        fullfile(qRepDir,'figures',['h_spiky_slices_' sprintf('%d',i-6) '.png'])  )  % spiky slices
    end
  end
end


%% save spikes file
% If there were any spikes, save the structure out.peaks into a .mat file.

if nFigs>6  % i.e., if there were any spikes detected
  % save the field 'peaks' of the structure 'out':
  save( fullfile(qRepDir,[name '_spikes.mat']), '-struct', 'out', 'peaks');
end


%%
  %%%%%%%%%%%%%%%%%%%%%%%%
  %  Write HTML Report   %
  %%%%%%%%%%%%%%%%%%%%%%%%

% Open html file:
fid = fopen( fullfile(qRepDir,[qRepRoot '.html']), 'w' );

fprintf(fid,'<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">\n');
fprintf(fid,'<html>\n');
fprintf(fid,'<head>\n');
fprintf(fid,'  <meta content="text/html; charset=ISO-8859-1" http-equiv="content-type">\n');
fprintf(fid,'  <title>Data Quality Report - %s</title>\n', [name, ext]);
fprintf(fid,'</head>\n');
fprintf(fid,'\n');
fprintf(fid,'<body>\n');
fprintf(fid,'<h1>Data Quality Report - %s</h1>\n', [name, ext]);
fprintf(fid,'<br>\n');
fprintf(fid,'<big style="font-weight: bold;">\n');
fprintf(fid,'  Coil: %s<br>\n',out.coil);
fprintf(fid,'  Date: %s<br>\n',out.sDate);
fprintf(fid,'</big><br>\n');
fprintf(fid,'\n');
fprintf(fid,'\n');
fprintf(fid,'<hr style="width: 100%%; height: 2px;">\n');
fprintf(fid,'\n');

    %%%   ROI locations   %%%

% fprintf(fid,'<center>\n');
% fprintf(fid,'<table border = "0" width = "100%%">\n');
% fprintf(fid,'  <tr><td colspan = "2" align="center"><h3>ROI locations for slice in 1st ROI</h3></td>\n');
% fprintf(fid,'  <tr>\n');
% fprintf(fid,'    <td colspan = "2" align="center">\n');
% fprintf(fid,'      <table border = "0" width = "100%%">\n');
% fprintf(fid,'\n');
% fprintf(fid,'        <tr>\n');
% fprintf(fid,'          <td style = "width: 47px;font-weight: bold; color:red">red:</td>\n');
% fprintf(fid,'          <td>20x20 ROI in the center of the phantom, <b>PHANTOM MAXROI</b></td>\n');
% fprintf(fid,'          <td rowspan = "2">\n');
fprintf(fid,'            <img style = "width: 800px;" alt="ROI positions" src="%s"> \n',...
                                                      fullfile('figures','h_roi_loc.png'));
fprintf(fid,'          </td>\n');
% fprintf(fid,'          <td style="width: 47px;font-weight: bold; color:blue">blue:</td>\n');
% fprintf(fid,'          <td>\n');
% fprintf(fid,'            10x10 ROI in the part of the phantom that is generating the ghost signal in the green ROI, <b>PHANTOM</b>\n');
% fprintf(fid,'          </td>  \n');
% fprintf(fid,'        </tr>\n');
% fprintf(fid,'        <tr>\n');
% fprintf(fid,'            <td style="width: 47px;font-weight: bold; color:green">green:</td>\n');
% fprintf(fid,'            <td>2 10x5 ROIs in the part of the ghost with higher intensity, <b>GHOST</b></td>\n');
% fprintf(fid,'            <td style="width: 47px;font-weight: bold; color:yellow">yellow:</td>\n');
% fprintf(fid,'            <td >5x20 ROI in the background (equivalent to a 10x10 size), <b>BACKGROUND</b></td>\n');
% fprintf(fid,'        </tr>\n');
% fprintf(fid,'      </table>\n');
% fprintf(fid,'    </td>\n');
% fprintf(fid,'  </tr>\n');
fprintf(fid,'\n');
fprintf(fid,'<hr style="width: 100%%; height: 2px;">\n');
fprintf(fid,'\n');

    %%%   Different SNR indicators   %%%

fprintf(fid,'  <tr><td colspan = "2"  align = "center">\n');
fprintf(fid,'    <h3>SNR "indicators"</h3>\n');
fprintf(fid,'  </td></tr>\n');
fprintf(fid,'  <tr><td colspan = "2"  align = "center">\n');
fprintf(fid,'SNR_0 = %.2f\n', out.snr0);
fprintf(fid,'    <br>\n');
fprintf(fid,'SFNR  = %.2f\n', out.sfnr);
fprintf(fid,'    <br>\n');
fprintf(fid,'SGR   = %.2f\n', out.sgr);
fprintf(fid,'    <br>\n');
fprintf(fid,'<hr style="width: 100%%; height: 2px;">\n');
fprintf(fid,'\n');


  %%%   Intensity time-series at different locations   %%%

fprintf(fid,'  <tr><td colspan = "2"  align = "center">\n');
fprintf(fid,'    <h3>Intensity time-series at different locations</h3>\n');
fprintf(fid,'  </td></tr>\n');

fprintf(fid,'  <tr>\n');
fprintf(fid,'    <td align = "center">\n');
fprintf(fid,'      <b>- Object -</b>\n');
fprintf(fid,'    </td>\n');
fprintf(fid,'    <br>\n');
fprintf(fid,'    <td align = "center">\n');
fprintf(fid,'      <a href="%s"> \n', fullfile('figures','h_m_obj.png'));
fprintf(fid,'      <img style = "width: 500px; border: none" src="%s"  alt="Mean Intensity Objects ROI">\n', ...
                                      fullfile('figures','h_m_obj.png'));
fprintf(fid,'      </a>\n');
fprintf(fid,'    </td>\n');
fprintf(fid,'    <br>\n');

fprintf(fid,'    <td align = "center">\n');
fprintf(fid,'      <b>- Background -</b>\n');
fprintf(fid,'    </td>\n');
fprintf(fid,'    <br>\n');
fprintf(fid,'    <td align = "center">\n');
fprintf(fid,'      <a href="%s"> \n', fullfile('figures','h_m_bkg.png'));
fprintf(fid,'      <img style = "width: 500px; border: none" src="%s"  alt="Mean Intensity Background ROI">\n', ...
                                      fullfile('figures','h_m_bkg.png'));
fprintf(fid,'      </a>\n');
fprintf(fid,'    </td>\n');
fprintf(fid,'    <br>\n');

fprintf(fid,'    <td align = "center">\n');
fprintf(fid,'      <b>- Ghost -</b>\n');
fprintf(fid,'    </td>\n');
fprintf(fid,'    <br>\n');
fprintf(fid,'    <td align = "center">\n');
fprintf(fid,'      <a href="%s"> \n', fullfile('figures','h_m_gho.png'));
fprintf(fid,'      <img style = "width: 500px; border: none" src="%s"  alt="Mean Intensity Ghost ROI">\n', ...
                                      fullfile('figures','h_m_gho.png'));
fprintf(fid,'      </a>\n');
fprintf(fid,'    </td>\n');
fprintf(fid,'  </tr>\n');
fprintf(fid,'\n');


    %%%   Spikes   %%%

fprintf(fid,'  <tr><td colspan = "2" align = "center"><br><hr style="height: 2px; width: 100%%;"><br></td></tr>\n');
fprintf(fid,'\n');
fprintf(fid,'  <tr><td colspan = "2"  align = "center">\n');
fprintf(fid,'    <h3>Spike Detection Report</h3>\n');
fprintf(fid,'  </td></tr>\n');
fprintf(fid,'  <tr><td colspan = "2"  align = "center">\n');

if nFigs>5
%   fprintf(fid,'  <tr>\n');
  fprintf(fid,'    Number of occurrences of potential spikes: %d\n',sum([out.peaks.nspk]));
  fprintf(fid,'    <br>\n');
  fprintf(fid,'    <br>\n');
  for n=1:length(out.peaks)
    if out.peaks(n).nspk      % if there are any spikes for this ROI
      % We print on screen where they are:
      fprintf(fid,'Spikes ROI: %s\n',out.peaks(n).name );
      fprintf(fid,'    <br>\n');
      for i = 1:length(out.peaks(n).ts)
        fprintf(fid,'t = %d\n', out.peaks(n).ts(i));
        fprintf(fid,'    <br>\n');
      end
    end
  end
  fprintf(fid,'    <br>\n');

%   fprintf(fid,'  </td></tr>\n');
%   fprintf(fid,'  <tr><td colspan = "2" align = "center">\n');
  fprintf(fid,'    <b>- Mean intensity Spikes ROIs -</b>\n');
  fprintf(fid,'    <br>\n');
  fprintf(fid,'    <a href="%s"> \n', fullfile('figures','h_spikes.png'));
  fprintf(fid,'    <img style = "width: 500px; border: none" src="%s"  alt="Mean Intensity Background, for spike detection.">\n', ...
                                      fullfile('figures','h_spikes.png'));
  fprintf(fid,'    </a>\n');
  fprintf(fid,'    <br>\n');
  if nFigs>6
    for i=7:nFigs
      fprintf(fid,'    <a href="%s"> \n', fullfile('figures',['h_spiky_slices_' sprintf('%d',i-6) '.png']));
      fprintf(fid,'    <img style = "width: 500px; border: none" src="%s"  alt="Mean Intensity Background, for spike detection.">\n', ...
         fullfile('figures',['h_spiky_slices_' sprintf('%d',i-6) '.png']));
      fprintf(fid,'    </a>\n');
      fprintf(fid,'    <br>\n');
    end
  end
else
  fprintf(fid,'    Insufficient number of repetitions for spike analysis.\n');
  fprintf(fid,'    <br>\n');
end
fprintf(fid,'  </td></tr>\n');
fprintf(fid,'\n');
fprintf(fid,'</center>\n');
fprintf(fid,'</body>\n');
fprintf(fid,'</html>\n');

fclose(fid);


%% Clean up

% close all open figures
close(out.handles);
% since the figures don't exist any longer, remove handles from output
out = rmfield(out,'handles');

% copy the ROI file
statement = sprintf('! cp %s %s', roifName, qRepDir);
eval(statement)
