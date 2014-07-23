function [data,cbihdr] = cbiReadNifti(fname,subset,prec,short_nan)
% [data,hdr]=cbiReadNifti(fname,subset,prec,short_nan)
% 
% Loads all or part of a Nifti-1 file.
%  Default is to load entire file as a double precision array
%  with dimensions [xSize ySize zSize tSize] where x,y,z,t are
%  defined according to the Nifti standard (x fastest changing 
%  dimension, y second and so on - usually x corresponds to left-right,
%  y to back-front, z to down-up, and t to early-late. 
% 
% hdr contains the NIfTI header as defined in @niftifile
% 
% Options:
%  subset:    4x1 cell array describing image subset to retrieve. 1-offset (Matlab-style).
%             Examples: 
%             - to retrieve a single z-slice (e.g. 4):  subset={[],[],4,[]}
%             - to retrieve a single voxel time series (e.g. [5,6,7]):  subset={5,6,7,[]}
%             - to retrieve a single volume from a time series (e.g. second volume):  subset={[],[],[],2}
%             - to retrieve a block of voxels from a volume: eg. subset={[4 50],[6 20],[1 10],[]}
%             If data is quasi-5D, subset{4} defines the subset in dim(5)
% 
%  prec:      Storage type of returned data. Legal values are:
%             'native' - data are returned as stored in file (no scaling). Only works for loading
%              contiguous data (entire volumes).
%             'double' - data are returned in Matlab standard double precision format (default)
%  short_nan: NaN handling for signed short (int16) data. If 1, will treat -32768 (smallest 
%             representable number) as NaN, reserving -32767..32767 for scaled data; otherwise
%             treats -32786 as a real value. Default is 1. 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                                             %%
%% Note: this function follows the same format of the cbiReadNifti function part of mrTools.   %%
%%                                                                                             %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if nargin < 1
    help cbiReadNifti;
    return
  end
  
  if ~exist('prec', 'var')
    prec = 'double';
  end
  if ~exist('short_nan', 'var')
    short_nan = 1;
  end

  hdr = niftifile(fname);
  hdr = fopen(hdr, 'read');
  if ~exist('subset', 'var')
    specDims = hdr.dim(1);
    for d = 1:min(specDims, 4)
      subset{d} = [int32(1) hdr.dim(d+1)];
    end
    for d = specDims+1:4
      subset{d} = [int32(1) int32(1)];
    end
  else
    % Loop through subset and verify limits
    for nsub = 1:numel(subset)
      if numel(subset{nsub}) > 0
        if (subset{nsub}(1) < 1 || subset{nsub}(1) > hdr.dim(nsub+1)) ...
              || (numel(subset{nsub}) == 2 && (subset{nsub}(2) < 1 || subset{nsub}(2) > hdr.dim(nsub+1)))
          error('Illegal range "', mat2str(subset{nsub}), '"');
        end
      end
    end
    % Append possibly missing ranges
    for asub = nsub+1:4
      subset{asub} = [1 hdr.dim(asub+1)];
    end
  end
  [hdr, data, size] = fread(hdr, subset{1}, subset{2}, subset{3}, subset{4}); 
  hdr = fclose(hdr);

  % Scaling
  if strcmp(prec, 'double') && (hdr.scl_slope ~= 0)  
    % NaN handling for int16 data
    if strcmp(hdr.datatype, 'int16') && short_nan 
      data(data == -32768) = NaN;
    end
    % Use double precision version of slope and inter to satisfy Matlab's type checking
    data = double(data).*double(hdr.scl_slope)+double(hdr.scl_inter);
  end

  % 1D curvature file fix: if the width is at maximum, we should reshape it to a 1xn image
  maxWidth = 2^15-1;
  if (hdr.dim(3) == maxWidth)
    % check to see if the description has the real width
    [realWidthStart realWidthEnd] = regexp(hdr.descrip,'^[0-9]*:');
    if ~isempty(realWidthStart)
      % get the real width
      realWidth = str2num(hdr.descrip(realWidthStart:realWidthEnd-1));
      % reset the description
      hdr.descrip = hdr.descrip(realWidthEnd+1:end);
      % tell user what we are doing
      disp(sprintf(['(cbiReadNifti) Fixing dimensions of image written as %ix%i becasue of nifti ' ...
                    'format limitations but actually is 1x%i'],hdr.dim(2),hdr.dim(3),realWidth));
      
      % fix the header
      hdr.dim(2) = 1;
      hdr.dim(3) = realWidth;
      % fix the data
      data = data(1:realWidth);
    elseif (hdr.dim(2)==1)
      mrErrorDlg(sprintf(['(cbiReadNifti) %s is an old style surface curvature file. You need to ' ...
                          're-import the surface'],fname));
    end
  else
    % Reshape the data
    datashape = zeros(4,1);
    for n = 1:4
      if isempty(subset{n})
        datashape(n) = hdr.dim(n+1);
      elseif numel(subset{n}) == 1
        datashape(n) = 1;
      else
        datashape(n) = subset{n}(2) - subset{n}(1) + 1;
      end
    end
    data = reshape(data, datashape');
  end

  cbihdr = cbiCreateNiftiHeader(hdr);
  