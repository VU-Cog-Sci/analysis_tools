function out = checkdataquality(datafName,roifName,...
                                  tExclude, defFontSize, defLineWidth, defMarkerSize)

% checkdataquality
%
%     out = checkdataquality(datafName,roifName)
%
%     out = checkdataquality(datafName,roifName,...
%                                tExclude, defFontSize, defLineWidth, defMarkerSize)
%
% Function to check for data quality in a fMRI series:
%
% It reads the data stored in 'datafName'.  (It works both for nifti and
%   dicom).  It calculates the mean & standard deviation (std) of the data
%   at a bunch of different ROIs defined in roifName (they are supposed to
%   be in the object, in the ghost and in the background).  Then, it plots
%   (and saves the image) the time courses for some of the ROIs, and
%   calculate some statistics, to review the quality of the data.
% Use 'roiCorners' to generate the roifName.
% 
% The output structure contains:
%   - snr0
%   - sfnr
%   - sgr
%   - sDate: scan date
%   - coil: coil used for the scan
%   - peaks
%   - handles (to figures)
%
% If your images are NIfTI, you need to have in your matlab path the
%   directory with the matlab-nifti functions; as well as 'getNSlices.m'
%
% By passing the optional argument tExclude you can exclude the first few
%   timepoints of the series.  By passing defFontSize, defLineWidth and
%   defMarkerSize you can change the appearance of the figures.
%
% PJV: v.3.4: Jan 2014

% PJV: v.3.5: Jan 2014
%   It doesn't call CBIrobustfit when the time-series is constant
%   (otherwise it prints a ton of warnings about the matrix being
%   singular).
% PJV: v.3.4: April 2012
%   Allows the user to exclude a few time-points at the beginning of the
%   run.
% PJV: v.3.3: February 2009
%   Bug fixed: it now gets Nx and Ny for DICOM from AcquisitionMatrix.
% PJV: v.3.2: February 2009
%   Bug fixed: when datafName included a folder, the path for the
%     '*-header.txt' file was wrong.
% PJV: v.3.1: January 2009
%   Uses de-trended time-course to determine size of noise fluctuations.
%   ROI importing and plotting section in separate functions.
% PJV: v.3.0: December 2008
%   Uses 'roi' structures, to make easy to pass variables around.
%   Reads the ROIs from the file in a separate function,
% PJV: v.2.0: October 2008
%   Reads the ROIs from a separate text file.
%   Improved handling of file types.
% PJV: v.1.3: October 2008
%   Fixed the bug of program breaking when data type = short.
% PJV: v.1.2: November 2006
%   Added new plot: central slice with most representative ROIs. It is
%     windowed to show the ghost.
%   It also fixes orientation problem with Nifti files.
% PJV: v.1.1: November 2006


%%   initialization   %%

%%% MODIFY this if you want a different value %%%
thold = 9;   % threshold for detecting spikes

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


%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%   Get data header  %%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%

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


%%   read the header   %%
switch filetype

  case 'DICOM'
    hdr = dicominfo(datafName);

    if strcmpi(hdr.InPlanePhaseEncodingDirection, 'ROW')
      % Phase-Encoding dimension: 1 = growing fastest in file (x); 2 = y;
%       PE_dim = 2;
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
%       PE_dim = 1;
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

    if bMosaic
      % when MOSAIC format, all slices for a given volume are in 1 file.
      file_list = dir([datafName(1:end-8), '*' ext]);
      Nt = length(file_list);                 % repetitions
    else    % no MOSAIC image
      % the file structure in the directory is more complicated: there are
      % Nz * Nt files:
      file_list = dir([datafName(1:end-14) '*' ext]);
      Nt = length(file_list)/Nz;
    end

  case 'Nifti'
    hdr = niftifile(datafName);
    hdr = fopen(hdr, 'read');

    Nz     = double(hdr.nz);
    Nx     = double(hdr.nx);
    Ny     = double(hdr.ny);
    Nt     = double(hdr.nt);
    % Phase-Encoding dimension: 1 = growing fastest in file (x); 2 = y;
%     PE_dim = hdr.phase_dim;

end

% I want Nx, Ny, Nz to be even.  If not, add 1.  That way, the center will
%    be Nx/2, Ny/2:
Nx = ceil(Nx/2)*2;   Ny = ceil(Ny/2)*2;

% Check that we haven't been asked to exclude more volumes than in the
%   data:
if (tExclude >= Nt)
  error('tExclude is too large.')
end

%% Get date and coil used from '*-header.txt' file:

my_header_file = dir( fullfile(pathstr,'*-header.txt') );
if ~isempty( my_header_file )
  [sDate, coil] = getDateCoil(  fullfile(pathstr,my_header_file.name)  );
else
  sDate = 'unknown';
  coil  = 'unknown';
end


%%
  %%%%%%%%%%%%%%%%%%%%
  %%%   Get ROIs   %%%
  %%%%%%%%%%%%%%%%%%%%

% % define (fields of an empty) structure to hold the ROIs:
% roi = struct('name',{}, ...
%              'type', {}, ...
%              'corner1', [], ...
%              'corner2',[], ...
%              'nvox',[], ...
%              'meants',[], ...
%              'stdts',[]);  
  
% Read the coordinates of the corners of the ROIs
[object,ghost,background,spikes] = importROIs(  roifName  );


%%
  %%%%%%%%%%%%%%%%%%%%%%%%%
  %%%   Read the data   %%%
  %%%%%%%%%%%%%%%%%%%%%%%%%

%   intitialize storage:
vol3D = zeros(Nx,Ny,Nz);
for n=1:length(object)
  object(n).meants = zeros(Nt-tExclude,1);
  object(n).stdts  = zeros(Nt-tExclude,1);
end
for n=1:length(ghost)
  ghost(n).meants = zeros(Nt-tExclude,1);
  ghost(n).stdts  = zeros(Nt-tExclude,1);
end
for n=1:length(background)
  background(n).meants = zeros(Nt-tExclude,1);
  background(n).stdts  = zeros(Nt-tExclude,1);
end
for n=1:length(spikes)
  spikes(n).meants = zeros(Nt-tExclude,1);
  spikes(n).stdts  = zeros(Nt-tExclude,1);
end


%   Read the data & store ROI statistics

% Excluded volumes:
if (tExclude>0)
  switch filetype
    case 'DICOM'
      % nothing to be done
    case 'Nifti'
      % skip those volumes
      hdr = fseek(hdr, tExclude*hdr.nx*hdr.ny*hdr.nz, 'cof');
  end   % of 'switch'
end      

for rep = tExclude+1:Nt   % loop over repetitions

  switch filetype
    case 'DICOM'

      if bMosaic  % image in MOSAIC format
        mosaic_ima = single( dicomread(fullfile(pathstr,file_list(rep).name)) );
        for z = 1:Nz                  % loop over slices
          vol3D(:, :, z) = mosaic_ima( ...
                   floor((z-1)/(double(hdr.Rows)/Nx))*Nx+(1:Nx),...
                     mod( z-1 ,(double(hdr.Columns)/Ny))*Ny+(1:Ny));
        end % of looping over slices
      else        % image not in MOSAIC: 1 slice per dicom file:
        for z = 1:Nz                  % loop over slices
          file_no = (rep-1)*Nz + z;
          vol3D(:, :, z) = single( dicomread( fullfile(pathstr,file_list(file_no).name) ) );
        end
      end    % of 'if bMosaic'

    case 'Nifti'
      [hdr, buff] = fread(hdr, hdr.nx*hdr.ny*hdr.nz);
      vol3D = single( reshape(buff, [hdr.nx, hdr.ny, hdr.nz]) );
  end   % of 'switch'

  for n=1:length(object)
    roiIntensity = vol3D( object(n).corner1(1):object(n).corner2(1), ...
                          object(n).corner1(2):object(n).corner2(2), ...
                          object(n).corner1(3):object(n).corner2(3) );
    object(n).meants(rep-tExclude) = mean( roiIntensity(:) );
    object(n).stdts(rep-tExclude)  =  std( roiIntensity(:) );
  end
  for n=1:length(ghost)
    roiIntensity = vol3D( ghost(n).corner1(1):ghost(n).corner2(1), ...
                          ghost(n).corner1(2):ghost(n).corner2(2), ...
                          ghost(n).corner1(3):ghost(n).corner2(3) );
    ghost(n).meants(rep-tExclude) = mean( roiIntensity(:) );
    ghost(n).stdts(rep-tExclude)  =  std( roiIntensity(:) );
  end
  for n=1:length(background)
    roiIntensity = vol3D( background(n).corner1(1):background(n).corner2(1), ...
                          background(n).corner1(2):background(n).corner2(2), ...
                          background(n).corner1(3):background(n).corner2(3) );
    background(n).meants(rep-tExclude) = mean( roiIntensity(:) );
    background(n).stdts(rep-tExclude)  =  std( roiIntensity(:) );
  end
  for n=1:length(spikes)
    roiIntensity = vol3D( spikes(n).corner1(1):spikes(n).corner2(1), ...
                          spikes(n).corner1(2):spikes(n).corner2(2), ...
                          spikes(n).corner1(3):spikes(n).corner2(3) );
    spikes(n).meants(rep-tExclude) = mean( roiIntensity(:) );
    spikes(n).stdts(rep-tExclude)  =  std( roiIntensity(:) );
  end

end % of looping over repetitions


%%  
  %%%%%%%%%%%%%%%%%%%%%
  %%%   Plot ROIs   %%%
  %%%%%%%%%%%%%%%%%%%%%
  
h_roi_loc = displayROIs(datafName,roifName,defFontSize, defLineWidth, defMarkerSize);


%%
  %%%%%%%%%%%%%%%%%%%%%%%%
  %%%   SNR Analysis   %%%
  %%%%%%%%%%%%%%%%%%%%%%%%  
  
% snr0 = mean(m_c20)  / (1.53*mean(s_c20)); % SNR from [Weisskoff 1996]
% sfnr = mean(m_c20)  / std(m_c20);         % S-to-fluctuation [Glover 2004]
% sgr  = mean(m_bg10) / mean(m_g10);        % S-to-ghost-ratio [Simmons 1999]

%% calculate the mean intensity for each of the regions:
% we go through the rois for each (not for spikes) and compute the mean and
%   std.
% NOTE: I'm not taking into account repeated voxels (if rois overlap).

%%% object:
cumsum_inten_obj = 0;
% cumsum_dev2_obj  = 0;
n_vox_obj = 0;
cumsum_inten_obj_gho = 0;
n_vox_obj_gho = 0;
% for the "object" regions, I also want the mean intensity in the ROIs
% linked to the ghost.
for n=1:length(object)
  cumsum_inten_obj = cumsum_inten_obj + object(n).meants*object(n).nvox;
%   cumsum_dev2_obj = cumsum_dev2_obj + (object(n).stdts).^2*object(n).nvox;
  n_vox_obj = n_vox_obj + object(n).nvox;
  % check if this ROI is linked to ghost (given in 'name' field):
  if strfind( lower(object(n).name), 'ghost' )
    cumsum_inten_obj_gho = cumsum_inten_obj_gho + object(n).meants*object(n).nvox;
    n_vox_obj_gho = n_vox_obj_gho + object(n).nvox;
  end
end
m_inten_obj = cumsum_inten_obj / n_vox_obj;       % mean across space
% s_inten_obj = sqrt(cumsum_dev2_obj / n_vox_obj);  % std  across space

m_inten_obj_gho = cumsum_inten_obj_gho / n_vox_obj_gho;
if (cumsum_inten_obj_gho==0)
  % this means we didn't find any ROI linked to the ghost:
  warning(['No ROI in the OBJECT was linked to the ghost.\n' ...
           'Signal-to-ghost ratio will be 0.\n' ...
           'Please check %s\n'], roifName);
end

%%% ghost:
cumsum_inten_gho = 0;
% cumsum_dev2_gho = 0;
n_vox_gho = 0;
for n=1:length(ghost)
  cumsum_inten_gho = cumsum_inten_gho + ghost(n).meants*ghost(n).nvox;
%   cumsum_dev2_gho = cumsum_dev2_gho + (ghost(n).stdts).^2*ghost(n).nvox;
  n_vox_gho = n_vox_gho + ghost(n).nvox;
end
m_inten_gho = cumsum_inten_gho / n_vox_gho;
% s_inten_gho = sqrt(cumsum_dev2_gho / n_vox_gho);


%%% background:
cumsum_inten_bkg = 0;
cumsum_dev2_bkg  = 0;
n_vox_bkg = 0;
for n=1:length(background)
  cumsum_inten_bkg = cumsum_inten_bkg + background(n).meants*background(n).nvox;
  cumsum_dev2_bkg = cumsum_dev2_bkg + (background(n).stdts).^2*background(n).nvox;
  n_vox_bkg = n_vox_bkg + background(n).nvox;
end
m_inten_bkg = cumsum_inten_bkg / n_vox_bkg;
s_inten_bkg = sqrt(cumsum_dev2_bkg / n_vox_bkg);


%% de-trend the object intensity time-courses:

t = linspace(0,1,Nt-tExclude)';
X = [ones(Nt-tExclude,1), t, t.^2];
invX = pinv(X);
m_obj_detr = m_inten_obj - X*(invX*m_inten_obj);

%% compute different quality "indicators" (signal-to-noise)

out.snr0 = mean(m_inten_obj) / (1.53*mean(s_inten_bkg)); % SNR from [Weisskoff 1996]
out.sfnr = mean(m_inten_obj) / std(m_obj_detr);          % S-to-fluctuation [Glover 2004]
out.sgr  = mean(m_inten_obj_gho) / mean(m_inten_gho);    % S-to-ghost-ratio [Simmons 1999]

  % Print to screen (for the report) the title:
  fprintf('--- SNR RESULTS ---\n');
  fprintf('SNR_0 = %.2f\n', out.snr0);
  fprintf('SFNR  = %.2f\n', out.sfnr);
  fprintf('SGR   = %.2f\n', out.sgr);


%%  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%   Plot ROIs time-series   %%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
%%% Mean signal in significant ROIs: %%%  
h_m_obj = figure;
plot(m_inten_obj, 'r-*');
set(gca, 'FontSize', defFontSize);
title('Average intensity of ROIs in object');
ylabel('Average signal');   xlabel('Rep No.');
text(0.3, 0.85, sprintf('SNR_0 = %.2f', out.snr0), ...
                'Units', 'normalized', 'FontSize', defFontSize);

h_m_bkg = figure;
plot(m_inten_bkg, 'r-*');
set(gca, 'FontSize', defFontSize);
title('Average intensity of ROIs in background');
ylabel('Average signal'); xlabel('Rep No.')

h_m_gho = figure;
plot(m_inten_gho, 'r-*');
set(gca, 'FontSize', defFontSize);
title('Average intensity of ROIs in ghost');
ylabel('Average signal');   xlabel('Rep No.')

h_m_obj_gho = figure;
plot(m_inten_obj_gho, 'r-*');
set(gca, 'FontSize', defFontSize);
title('Average intensity of ROIs in object linked to ghost');
ylabel('Average signal');  xlabel('Rep No.')
text(0.3, 0.85, sprintf('Signal-to-Ghost Ratio = %.2f', out.sgr), ...
                'Units', 'normalized', 'FontSize', defFontSize);



%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%   search for spikes   %%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Search for peaks in intensity in a the ROIs in the 'spikes' section
%   of the ROI file.
% We search for peaks in each of the 'spikes' ROIs.  Peak is when the signal goes
%   above the [mean (across time) + threshold * std].
% We store the spike candidates in a structure called 'peaks', with fields:
%   -'ts': timepoints where a spike was detected (for that ROI)
%   -'nspk': number of spikes (again, for that ROI)
%   -'slices': slice(s) included in the ROI
%   -'name': name of the ROI

% Print to screen (for the report) the title:
fprintf('--- SPIKE DETECTION ---\n');

if (Nt>10)  % only if there are a few timepoints!

  % create structure array to hold info about spikes for each "spike" roi:
  [peaks(1:length(spikes)).ts]   = deal(0);  % time-points with peaks
  [peaks(1:length(spikes)).nspk] = deal(0);  % total no. time-points w peaks
  
  for n = 1:length(spikes)  % loop through spike ROIs:
    % calculate the mean (across time) and check if time-series is constant
    %   (it could be all zeros, for example):
    m_meants = mean(spikes(n).meants);
    
    if any(~(spikes(n).meants==m_meants))
      % if the time series is not constant:
    
      % calculate the mean (across time) using robustfit, so that we don't
      %   take into acount the spikes ...:
      m_meants = CBIrobustfit( ones(Nt-tExclude,1), spikes(n).meants );
      % ... and the same for the std:
      s_meants = sqrt( ...
        CBIrobustfit( ones(Nt-tExclude,1), (spikes(n).meants-m_meants).^2 ) );
      peaks(n).ts =     find( spikes(n).meants > m_meants+thold*s_meants )+tExclude;
      peaks(n).nspk =   length(peaks(n).ts);
      peaks(n).slices = spikes(n).corner1(3):spikes(n).corner2(3);
      peaks(n).name = spikes(n).name;
      if peaks(n).nspk      % if there are any spikes for this ROI
        % We print on screen where they are:
        fprintf('Slice: %d (in "Spikes ROI": %s)\n',...
                spikes(n).corner1(3),spikes(n).name );
        fprintf('t = %d\n', peaks(n).ts);
      end
    end
    
  end  % of loop through slices
  
  % if no spikes were found, print it on screen
  if ~any([peaks.nspk])
    fprintf('No spike candidates found');
  end

else  % if Nt<=10
  peaks = NaN;  % Spike candidates.
  warning( ['Only %d time-points found.\n' ...
            'No spike detection analysis performed'], Nt );

end   % of 'if (Nt>10)

fprintf('\n\n')


  %%%%%%%%%%%%%%%%%%%%%%%
  %%%   Plot Spikes   %%%
  %%%%%%%%%%%%%%%%%%%%%%%

% initialize handle for spikes figures:
h_spikes = [];
h_spiky_slices = [];

if (Nt>10)  % only if there are a few timepoints!

%   disp('Background Timeseries for each slice');
  
  %%% We plot the mean of the background ROI, for all slices, across time:
  h_spikes = figure;
  clf
  backgr_intensity = zeros(Nt-tExclude, length(spikes));
  for n=1:length(spikes)
    backgr_intensity(:,n) = spikes(n).meants;
  end
  plot(backgr_intensity);
  set(gca, 'FontSize', defFontSize);
  xlabel('Rep No.');  ylabel('mean signal in background ROI');
  title('Background Timeseries for Spike detection', 'FontWeight', 'Bold');

  %%% Plot slices with spikes candidates, near the noisy time-points %%%
  % do it only if there aren't too many (say, for example, 20):
  if sum([peaks.nspk])<=21
  
    ar = double(Nx)/double(Ny);     % aspect ratio (for plotting slices)

    h_spiky_slices = zeros(1,sum([peaks.nspk]));
    i=0;  % cummulative number of spikes across ROIs
    for n = 1:length(spikes)  % loop through spike ROIs:
      if peaks(n).nspk       % if there are any spikes for this ROI
        z = spikes(n).corner1(3);   % slice number
        for j = 1:peaks(n).nspk   % loop through the spikes in that ROI
          i=i+1;
          % we plot the adjacent slices:
          paperPos = [0.05 0.05 6.9 2.9];   % Paper position. Also for display
          h_spiky_slices(i) = figure;
          set(gcf,'PaperSize',[7 3],'PaperPosition',paperPos)
          paperAr  = paperPos(3)/paperPos(4);   % Paper aspect ratio
          splot_width = 0.175;  % from 0 to 1
          splot_height = splot_width*ar*paperAr;
          for t = 1:5
            tt = peaks(n).ts(j)-3+t;  % timepoint number ('peaks(z).ts(j)' is the noisy one)
            axes_position = [0.05 + splot_width*1.05*(t-1),   0.12, ...    % left, bottom
                             splot_width,       splot_height];         % width,height
            axes('position', axes_position)
            % make sure the timepoint(tt) we are going to plot is in the series
            if ( (tt>0) && (tt<=Nt) )
              %%% read the slice data %%%
              switch filetype
                case 'DICOM'
                  if bMosaic  % image in MOSAIC format
                    mosaic_ima = dicomread(fullfile(pathstr,file_list(tt).name));
                    slice(:, :) = mosaic_ima( ...
                         floor((z-1)/(double(hdr.Rows)/Nx))*Nx+(1:Nx),...
                           mod( z-1 ,(double(hdr.Columns)/Ny))*Ny+(1:Ny));
                  else        % image not in MOSAIC: 1 slice per dicom file:
                    file_no = (rep-1)*Nz + z;
                    slice(:, :) = dicomread( fullfile(pathstr,file_list(file_no).name) );
                  end    % of 'if bMosaic'
                case 'Nifti'
                  % skip (tt-1) repetitions & (z-1) slices
                  offset = (tt-1)*Nx*Ny*Nz + (z-1)*Nx*Ny;
                  hdr = fseek(hdr, offset, 'bof');
                  [hdr, data_1D] = fread(hdr, Nx*Ny);
                  slice = reshape(data_1D, [Nx, Ny]);
              end   % of 'switch'
              % we plot the slice, w/ scale 5*the average of the spike ROI:
              imagesc(squeeze(abs(double(slice))), ...
                              [0 5*spikes(n).meants( peaks(n).ts(j)-tExclude )]);
            else   % (from "if ( (tt>0) && ...  )
              imagesc(zeros( Nx,Ny ));
            end
            set(gca, 'FontSize', defFontSize);
            axis image;
            colormap gray;
            set(gca,'XTick',[],'YTick',[])      % No axes ticks or labels
            xlabel( sprintf('t = %d', tt) );
            if t==3 % middle graph
              title( sprintf('Slice: %d (in "Spikes ROI": %s)\n',...
                                z,spikes(n).name ), 'Interpreter','none' );
            elseif t==5   % rightmost graph
%               colorbar
            end   % if t==3...
          end     % for t =...
        end       % for j =...  (loop through spikes in that ROI)
      end         % if nspk(z)  (are there any spikes in this ROI)
    end           % of loop through slices

  else     % if there are too many spikes
    fprintf('Too many spike candidates.  We are not plotting each one');
  end      % of plotting all spikes
end             % of 'if (Nt>10)

if strcmp(filetype,'Nifti')
  fclose(hdr);
end

%% Prepare output fields that has not been included yet

% date and coil
out.sDate = sDate;
out.coil  = coil;

% add the 'peaks' structure to the output:
out.peaks = peaks;

% add the figure handles:
out.handles = [h_roi_loc, h_m_obj, h_m_bkg, h_m_gho, h_m_obj_gho, ...
               h_spikes, h_spiky_slices];


%%

% %%% "Weisskoff" plot (relative fluctuations): %%%
% if (Nt>10)  % only if there are a few timepoints!
% 
%   figure;
%   loglog(rr, F_n, '-x', rr, fcalc, '--');
%   grid
%   set(gca, 'FontSize', defFontSize);
%   xlabel('ROI full width, pixels');  ylabel('Relative std, %');
%   axis([r1 r2 .01 F_n(1)]);
%   legend('measured', 'calculated');
%   text(0.4, 0.85, sprintf('decorrelation distance = %3.1f pixels',rdc), ...
%                 'Units', 'normalized', 'FontSize', defFontSize);
% 
% end             % of 'if (Nt>10)


function [sDate, coil] = getDateCoil(  my_header_file  )
  
% open 'my_header_file', find the Study Date and Coil and return them.

% open 'my_header_file':
fid = fopen(my_header_file,'r');


%% Loop through all the lines:

b_sDate = false;
b_coil  = false;
while ~(b_sDate && b_coil)
  tline = fgetl(fid);
  if ~ischar(tline)   % it will read a -1 at the end-of-file
    break
  end
%   disp( tline );    % for debugging
  k = strfind( tline , 'Study Date' );
  if k>0
    sDate = tline( k+length('Study Date//') : end );
    b_sDate = true;
  end
  k = strfind( tline , 'Coil' );
  if k>0
    coil = tline( k+length('Coil//') : end );
    b_coil = true;
  end
end

if ~b_sDate
  sDate = 'unknown';
end
if ~b_coil
  coil  = 'unknown';
end

% close ROIs my_header_file:
fclose(fid);



