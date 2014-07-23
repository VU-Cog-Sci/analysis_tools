function [status] = roiCorners(datafName,roifName)

% roiCorners
%
%   [status] = roiCorners(datafName,roifName)
%
% Function to generate and display the coordinates of the corners of the
%   ROIs needed for checkingdataquality.m
%
% It reads the header for 'datafName'.  (Works both for NIfTI and DICOM).
%   Depending on the dimensions of the image it calculates the default
%   location of the corners of the different ROIs (planned to be in the
%   brain, in the ghost and in the background).  Then it calls
%   'displayROIs.m' to show where the slices are located.
%
% If a second argument is not passed, it uses the default 'rois.txt', in
%   the same folder as the data.
%
% If 'roifName' IS PASSED and does EXIST, it just displays the ROIs in it:
%   it doesn't generate any.
%
% Note that 'displayROIs' displays the location of all ROIs in the slice
%   given by 'corner1' of the 1st roi in the object (normally, the central
%   slice), so it is advisable to put all the ROIs (except for the 'spikes'
%   rois) in the same slice.
%
% It is highly recommended that you check the location of the default ROIs
%   to see that they are where they are supposed to be.  You can use the
%   image viewer of your choice for it.  If you find the ROIs are not
%   correctly defined, you can manually edit the output file with the ROI
%   corners.  Here is the format for the ROI file:
%   For each "section" (object, ghost, background and spikes) there are
%   several ROIs.  For each of them, I'm going to write the ROI explanation
%   followed by the coordinates of the corners in the following format:
%     ROI
%     i1, j1, k1   (indices for the first corner)
%     i2, j2, k2   (        for the oposite corner)
%   ('i' is the index changing fastest, then 'j', then 'z'; and they
%    start from 1, not from 0!)
%
% Dependencies:
%   - If your images are NIfTI, you need the CBI NIfTI-matlab libraries
%   - 'getNSlices.m'
%   - 'displayROIs.m'
%
% For DICOM files, pass the name of any of the files in the series (they
%   all have to be in the same directory).
%
% PJV: v.2.4: November 2011

% PJV: v.2.4: November 2011
%   Fixed a small bug in the labels of the ROIs.
% PJV: v.2.3: February 2009
%   Bug fixed: it now gets Nx and Ny for DICOM from AcquisitionMatrix.
% PJV: v.2.2: February 2009
%   - Default ROI file: 'rois.txt' in the same folder as the data.
%   - fixed a bug: if PE_dim is not properly set, use '2'.
% PJV: v.2.1: January 2009
%   - call 'displayROIs.m' to generate plot.
%   - if 'roifName' does exist, don't generate locations, just plot.
% PJV: v.2.0: December 2008
%   - 4 sections in ROI file: object, ghost, background & spikes
%   - Give the coordinates in 3D
% PJV: v.1.0: October 2008

status = 0;
bAutoGenerateROIs = true;    % by default, generate the ROIs automatically

%%   find type of file   %%

[pathstr, name, ext] = fileparts(datafName);

if isempty(ext)
  datafName = dir( fullfile( pathstr, [name '.*']) );
  if isempty(datafName)
    error('File not found.  Check the filename');
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

datafName = fullfile( pathstr, [name ext] );


%%   check 'roifName'   %%

% if no 'roifName' is passed (or blank), use default.
if ( (nargin<2) || isempty(roifName) )
  roifName = fullfile(pathstr,'rois.txt');
  bAutoGenerateROIs = true;
else   % if 'roifName' was specify by user:
  % if 'roifName' exists, do not generate the ROIs automatically
  %    (it will just display them)
  if exist(roifName, 'file')
    bAutoGenerateROIs = false;
  else
    bAutoGenerateROIs = true;
  end
end


%%

if bAutoGenerateROIs    % if we do have to generate them:

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%   Generate ROIs automatically   %%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  %%   read the header   %%
  switch filetype
    case 'DICOM'
      hdr = dicominfo(datafName);

      if strcmpi(hdr.InPlanePhaseEncodingDirection, 'ROW')
        % Phase-Encoding dimension: 1 = growing fastest in file (x); 2 = y;
        PE_dim = 2;
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
        % Phase-Encoding dimension: 1 = growing fastest in file (x); 2 = y;
        PE_dim = 1;
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

    case 'Nifti'
      hdr = niftifile(datafName);
      hdr = fopen(hdr, 'read');
      hdr = fclose(hdr);  % We opened it just to get the header.

      Nz     = double(hdr.nz);
      Nx     = double(hdr.nx);
      Ny     = double(hdr.ny);
      % Phase-Encoding dimension: 1 = growing fastest in file (i); 2 = j;
      PE_dim = double(hdr.phase_dim);
      % If it is not set, assume it is '2' and give a warning:
      if ~((PE_dim==1)||(PE_dim==2))
        PE_dim = 2;
        fprintf(['\nWarning: the Phase-Encoding direction was not' ...
                 'defined for the data.  Using default.\n' ...
                 'Double-check the GHOST ROIs in the plot!\n\n']);
      end
  end

  % I want Nx, Ny, Nz to be even.  If not, add 1.  That way, the center will
  %    be Nx/2, Ny/2:
  Nx = ceil(Nx/2)*2;   Ny = ceil(Ny/2)*2;

  % center slice:
  czn = ceil(Nz/2);

  %% Output stage %%

  % open output file for writing:
  fid = fopen( roifName, 'w' );
  if fid < 0
    status = fid;
    % print on screen some error message here
    error('could not open file for writing! \n')
  end


  % For each "section" (object, ghost, background and spikes) there are
  % several ROIs.  For each of them, I'm going to write the ROI explanation
  % followed by the coordinates of the corners in the following format:
  %     ROI
  %     i1, j1, k1   (indices for the first corner)
  %     i2, j2, k2   (        for the oposite corner)
  %   ('i' is the index changing fastest, then 'j', then 'z')


  %%%   OBJECT   %%%

  fprintf(fid, '=== OBJECT ===\n');
  fprintf(fid, '\n');

  % % In center of image (the number indicates the side size):
  % fprintf(fid, 'OBJECT_1: 1x1 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2, Ny/2, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2, Ny/2, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_2: 2x2 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2,   Ny/2, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+1, Ny/2+1, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_3: 3x3 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-1, Ny/2-1, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+1, Ny/2+1, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_4: 4x4 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-1, Ny/2-1, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+2, Ny/2+2, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_5: 5x5 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-2, Ny/2-2, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+2, Ny/2+2, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_6: 6x6 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-2, Ny/2-2, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+3, Ny/2+3, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_7: 7x7 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-3, Ny/2-3, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+3, Ny/2+3, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_8: 8x8 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-3, Ny/2-3, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+4, Ny/2+4, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_9: 9x9 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-4, Ny/2-4, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+4, Ny/2+4, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_10: 10x10 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-4, Ny/2-4, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+5, Ny/2+5, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_11: 11x11 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-5, Ny/2-5, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+5, Ny/2+5, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_12: 12x12 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-5, Ny/2-5, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+6, Ny/2+6, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_13: 13x13 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-6, Ny/2-6, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+6, Ny/2+6, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_14: 14x14 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-6, Ny/2-6, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+7, Ny/2+7, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_15: 15x15 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-7, Ny/2-7, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+7, Ny/2+7, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_16: 16x16 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-7, Ny/2-7, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+8, Ny/2+8, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_17: 17x17 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-8, Ny/2-8, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+8, Ny/2+8, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_18: 18x18 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-8, Ny/2-8, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+9, Ny/2+9, czn);
  % fprintf(fid, '\n');
  % %-
  % fprintf(fid, 'OBJECT_19: 19x19 center of Object\n');
  % fprintf(fid, '%d,%d,%d\n', Nx/2-9, Ny/2-9, czn);
  % fprintf(fid, '%d,%d,%d\n', Nx/2+9, Ny/2+9, czn);
  % fprintf(fid, '\n');
  % %-
  fprintf(fid, 'OBJECT_MAXROI: 20x20 center of Object\n');
  fprintf(fid, '%d,%d,%d\n', Nx/2-9,  Ny/2-9, czn);
  fprintf(fid, '%d,%d,%d\n', Nx/2+10, Ny/2+10, czn);
  fprintf(fid, '\n');
  %-
  % In the object, in the part corresponding to the ghost (see below), except
  % for gap:
  fprintf(fid, 'OBJECT_linked_to_Ghost: 10x10 in Object - region corresponding to 10x10 ROI in Ghost\n');
  if PE_dim == 1    % Phase encoding in 'x'
    fprintf(fid, '%d,%d,%d\n', Nx/2-4, floor(Ny/4)-4, czn);
    fprintf(fid, '%d,%d,%d\n', Nx/2+5, floor(Ny/4)+5, czn);
  elseif PE_dim == 2    % Phase encoding in 'y'
    fprintf(fid, '%d,%d,%d\n', floor(Nx/4)-4, Ny/2-4, czn);
    fprintf(fid, '%d,%d,%d\n', floor(Nx/4)+5, Ny/2+5, czn);
  end
  fprintf(fid, '\n');
  fprintf(fid, '\n');
  %-


  %%%   GHOST   %%%

  fprintf(fid, '=== GHOST ===\n');
  fprintf(fid, '\n');

  % In the ghost (in two parts: 1 at top of slice, 1 at bottom,
  %   centered at 1/4th of the image in readout dimension):
  fprintf(fid, 'GHOST: 10x10 in Ghost - 1st part: top of slice\n');
  if PE_dim == 1    % Phase encoding in 'x'
    fprintf(fid, '%d,%d,%d\n', Nx-4, floor(Ny/4)-4, czn); % 1/4 along RO direction
    fprintf(fid, '%d,%d,%d\n', Nx, floor(Ny/4)+5, czn);
  elseif PE_dim == 2    % Phase encoding in 'y'
    fprintf(fid, '%d,%d,%d\n', floor(Nx/4)-4, Ny-4, czn);
    fprintf(fid, '%d,%d,%d\n', floor(Nx/4)+5, Ny, czn);
  end
  fprintf(fid, '\n');
  %-
  fprintf(fid, 'GHOST: 10x10 in Ghost - 2nd part: bottom of slice\n');
  if PE_dim == 1    % Phase encoding in 'x'
    fprintf(fid, '%d,%d,%d\n', 2, floor(Ny/4)-4, czn);
    fprintf(fid, '%d,%d,%d\n', 6, floor(Ny/4)+5, czn);
  elseif PE_dim == 2    % Phase encoding in 'y'
    fprintf(fid, '%d,%d,%d\n', floor(Nx/4)-4, 2, czn);
    fprintf(fid, '%d,%d,%d\n', floor(Nx/4)+5, 6, czn);
  end
  fprintf(fid, '\n');
  fprintf(fid, '\n');
  %-


  %%%   BACKGROUND   %%%

  fprintf(fid, '=== BACKGROUND ===\n');
  fprintf(fid, '\n');

  % In the background (rectangular, one side of the slice):
  fprintf(fid, 'BACKGROUND: 4x50 in Background - (left side of slice)\n');
  fprintf(fid, '%d,%d,%d\n', 2, 2, czn);  % (the (2,2) corner is common to both PE choices
  % the 2nd corner stretches along the PE direction, so that we don't get
  % into the ghost:
  if PE_dim == 1    % Phase encoding in 'x' (I'm assuming Nx is at least 51!)
    fprintf(fid, '%d,%d,%d\n', 51, 5, czn);
  elseif PE_dim == 2    % Phase encoding in 'y' (I'm assuming Ny is at least 51!)
    fprintf(fid, '%d,%d,%d\n', 5, 51, czn);
  end
  fprintf(fid, '\n');
  %-
  % In the background, 2nd ROI (rectangular, other side of the slice):
  fprintf(fid, 'BACKGROUND: 4x50 in Background - (right side of slice)\n');
  if PE_dim == 1    % Phase encoding in 'x' (I'm assuming Nx is at least 51!)
    fprintf(fid, '%d,%d,%d\n', 2,  Ny-4, czn);
    fprintf(fid, '%d,%d,%d\n', 51, Ny-1, czn);
  elseif PE_dim == 2    % Phase encoding in 'y' (I'm assuming Ny is at least 51!)
    fprintf(fid, '%d,%d,%d\n', Nx-4, 2, czn);
    fprintf(fid, '%d,%d,%d\n', Nx-1, 51, czn);
  end
  fprintf(fid, '\n');
  fprintf(fid, '\n');
  %-


  %%%   SPIKES (in background, to search for spikes)   %%%

  fprintf(fid, '=== SPIKES (in background, to search for spikes) ===\n');
  fprintf(fid, '\n');

  % Small (5x10) in the background, to detect spikes; 1 per slice:
  for zn = 1:Nz
    fprintf(fid, ['BACKGROUND_SPIKES_SLICE_%d: 5x10 in Background - ', ...
      '(left side of slice)\n'], zn);
    fprintf(fid, '%d,%d,%d\n', 2, 2, zn); % (the (2,2) corner is common to both PE choices
    % the 2nd corner stretches along the PE direction, so that we don't get
    % into the ghost:
    if PE_dim == 1    % Phase encoding in 'x'
      fprintf(fid, '%d,%d,%d\n', 11, 5, zn);
    elseif PE_dim == 2    % Phase encoding in 'y'
      fprintf(fid, '%d,%d,%d\n', 5, 11, zn);
    end
    fprintf(fid, '\n');
  end
  %-


  %%% Close file %%%
  status = fclose(fid);

  
  
else    % if we don't have to generate them, do nothing: 
  % we'll read them from 'roifName'
end

%%  
  %%%%%%%%%%%%%%%%%%%%%
  %%%   Plot ROIs   %%%
  %%%%%%%%%%%%%%%%%%%%%
  

h_roi_loc = displayROIs(datafName,roifName);

if h_roi_loc<0
  status = 1;   % there was an error
end
