function h_roi_loc = displayROIs(datafName,roifName,...
                                  defFontSize, defLineWidth, defMarkerSize)

% displayROIs
%
%   h_roi_loc = displayROIs(datafName,roifName)
%   h_roi_loc = displayROIs(datafName,roifName,...
%                                 defFontSize, defLineWidth, defMarkerSize)
%
% Function to display the location of the ROIs defined in 'roifName' on the
%   image 'datafName'.
%
% It displays the location of all ROIs in the slice given by 'corner1' of
%   the 1st roi in the object (normally, the central slice), so it is
%   advisable to put all the ROIs (except for the 'spikes' rois) in the
%   same slice.  It returns the handle of the figure created.
%
% It reads only 1 volume from 'datafName'.  (Works both for NIfTI and
%   DICOM).  For DICOM files, pass the name of any of the files in the
%   series (they all have to be in the same directory).
%
% By passing the optional arguments defFontSize, defLineWidth and
%   defMarkerSize you can change the appearance of the figures.
%
% Dependencies:
%   - If your images are NIfTI, you need the CBI NIfTI-matlab libraries
%   - 'getNSlices.m'
%
% PJV: v.1.1: February 2009

% PJV: v.1.1: February 2009
%   Bug fixed: it now gets Nx and Ny for DICOM from AcquisitionMatrix.
% PJV: v.1.0: January 2009
%   created, from 'checkdataquality.m'

h_roi_loc = 0;

if (nargin<5)
  defMarkerSize = 10;     % Default marker size (for figures)
  if (nargin<4)
    defLineWidth = 1.2;   % Default line width (for figures)
    if (nargin<3)
      defFontSize = 12;   % Default font size (for labels)
    end
  end
end

%%   check arguments, and find type of file   %%

[pathstr, name, ext] = fileparts(datafName);

if isempty(ext)
  datafName = dir( fullfile( pathstr, [name '.*']) );
  if isempty(datafName)
    error('Data file not found.  Check the filename');
  end
  [pathstr, name, ext] = fileparts(datafName(1).name);
end

if strcmpi( ext, '.dcm')
  filetype = 'DICOM';
elseif (strcmpi( ext, '.nii') || ...
        strcmpi( ext, '.hdr') || ...
        strcmpi( ext, '.img') )
  filetype = 'Nifti';
else
  error('Unknown file extension.  The file must be "DICOM" or "Nifti"')
end

% if no 'roifName' is passed (or blank), return error.
if ( (nargin<2) || isempty(roifName) )
  error('You need to specify a "roifName".');
end
% if 'roifName' doesn't exist, return error:
if ~exist(roifName, 'file')
  error('ROI File not found.  Check the filename or run "roiCorners.m"');
end

datafName = fullfile( pathstr, [name ext] );



%%   read the header   %%

switch filetype

  case 'DICOM'
    hdr = dicominfo(datafName);
    
    if strcmpi(hdr.InPlanePhaseEncodingDirection, 'ROW')
      % AcquisitionMatrix: freq_rows\freq_columns\phase_rows\phase_columns
      Nx = double(hdr.AcquisitionMatrix(2));    % 'x' (i.e., 1st dimension, is freq.)
      Ny = double(hdr.AcquisitionMatrix(3));    % 'y' (i.e., 2nd dimension, is phase)
      % if PE dir is along 'ROW', the number of columns should be Ny
      % (y=2nd index); otherwise, the image is in MOSAIC format.
      if (hdr.Columns~=Ny)
        bMosaic = true;
      else
        bMosaic = false;
      end

    elseif strcmpi(hdr.InPlanePhaseEncodingDirection, 'COL')
      % AcquisitionMatrix: freq_rows\freq_columns\phase_rows\phase_columns
      Nx = double(hdr.AcquisitionMatrix(1));    % 'x' (i.e., 1st dimension, is freq.)
      Ny = double(hdr.AcquisitionMatrix(4));    % 'y' (i.e., 2nd dimension, is phase)
      % if PE dir is along 'COL', the number of rows should be Nx;
      % otherwise, the image is in MOSAIC format.
      if (hdr.Rows~=Nx)
        bMosaic = true;
      else
        bMosaic = false;
      end

    end

    Nz = double(getNSlices(hdr));           % no. slices

    % structure containing a list of all .dcm files in the same folder as
    % the input data file:
    file_list = dir(fullfile( pathstr,['*',ext] ));

  case 'Nifti'

    hdr = niftifile(datafName);
    hdr = fopen(hdr, 'read');

    Nz     = double(hdr.nz);
    Nx     = double(hdr.nx);
    Ny     = double(hdr.ny);

end

% I want Nx, Ny, Nz to be even.  If not, add 1.  That way, the center will
%    be Nx/2, Ny/2:
Nx = ceil(Nx/2)*2;   Ny = ceil(Ny/2)*2;



%% Import ROIs locations

[object,ghost,background,spikes] = importROIs(roifName);  



%% Read the data

vol3D = zeros(Nx,Ny,Nz,'single');

switch filetype
  case 'DICOM'
    if bMosaic  % image in MOSAIC format
      mosaic_ima = single( dicomread(datafName) );
      for z = 1:Nz        % loop over slices
        vol3D(:, :, z) = mosaic_ima( ...
                   floor((z-1)/(double(hdr.Rows)/Nx))*Nx+(1:Nx),...
                     mod( z-1 ,(double(hdr.Columns)/Ny))*Ny+(1:Ny));
      end % of looping over slices
    else        % image not in MOSAIC: 1 slice per dicom file:
      for z = 1:Nz                  % loop over slices
        rep = 1;    % first repetition
        file_no = (rep-1)*Nz + z;
        vol3D(:, :, z) = single( dicomread( fullfile(pathstr,file_list(file_no).name) ) );
      end
    end    % of 'if bMosaic'

  case 'Nifti'
    [hdr, buff] = fread(hdr, hdr.nx*hdr.ny*hdr.nz);
    vol3D = single( reshape(buff, [hdr.nx, hdr.ny, hdr.nz]) );
end   % of 'switch'



%%  
  %%%%%%%%%%%%%%%%%%%%%
  %%%   Plot ROIs   %%%
  %%%%%%%%%%%%%%%%%%%%%
  
  % Slice w/ ROI positioning %

% I'm going to plot the slice given by 'corner1' of the 1st roi in the
% object, and the different ROIs in it.
% This will allow the user to see if the ROIs are in the correct place.
% I get the handlers for the 1st "plot" of each ROI (for the legend)
% (NOTE: the indexing convention for 'imagesc' and 'plot' is reversed. That
%        is why in all the 'plot' commands, I use first the second
%        coordinate of the ROI corners: it is the one corresponding to the
%        horizontal axis .)
zn = object(1).corner1(3);

% global ROI array (I use it to make code "neater"):
roi = [object, ghost, background, spikes];

h_roi_loc = figure;

% set(gcf,'DefaultLineLineWidth',defLineWidth)
set(gcf,'PaperSize',[7 4.5],'PaperPosition',[0.1 0.1 6.9 4.4])

axes_position = [0.05, 0.1, ...    % left, bottom
                 0.6, 0.8];         % width,height
axes('position', axes_position)
set(gca, 'FontSize', defFontSize);
lineColors = 'rgbcy';
lineStyles = ['-';':'];
imagesc(abs(vol3D(:,:,zn)), [0 0.2*max(max(abs(vol3D(:,:,zn))))]);
hold on;
axis image;
colormap gray; colorbar;

%% loop through all rois, and find those in slice 'zn':

i=0;  % counter of number of rois displayed in this figure
for n=1:length(roi)
  % check whether this roi includes slice 'zn':
  if find( (roi(n).corner1(3):roi(n).corner2(3)) == zn )
    i=i+1;
    thisRoi = [ roi(n).corner1(1:2); roi(n).corner2(1:2) ];
    h(i) = plot(thisRoi(:,2), [thisRoi(1,1) thisRoi(1,1)], ...
      [lineColors( mod(i-1,length(lineColors))+1 ), ...
       lineStyles( ceil(i/length(lineColors)) )], ...
      'LineWidth', 2);
    plot(thisRoi(:,2), [thisRoi(2,1) thisRoi(2,1)], ...
      [lineColors( mod(i-1,length(lineColors))+1 ), ...
       lineStyles( ceil(i/length(lineColors)) )], ...
      'LineWidth', 2);
    plot([thisRoi(2,2) thisRoi(2,2)], thisRoi(:,1), ...
      [lineColors( mod(i-1,length(lineColors))+1 ), ...
       lineStyles( ceil(i/length(lineColors)) )], ...
      'LineWidth', 2);
    plot([thisRoi(1,2) thisRoi(1,2)], thisRoi(:,1), ...
      [lineColors( mod(i-1,length(lineColors))+1 ), ...
       lineStyles( ceil(i/length(lineColors)) )], ...
      'LineWidth', 2);
    
    leg_text{i} = lower( roi(n).name );
  end
end
% legend:
legend( h, leg_text, 'Interpreter','none', 'Location',[0.7 0.4 0.2 0.18]);
%         'Location', 'NorthOutside');
      
title('ROI locations -- slice in 1st ROI  ', 'FontWeight', 'Bold');
hold off;
