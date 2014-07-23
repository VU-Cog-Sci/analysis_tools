function [byteswritten,cbihdr] = cbiWriteNifti(fname,data,cbihdr,prec,subset,short_nan);
% [byteswritten,hdr] = cbiWriteNifti(filename,data,hdr,prec,subset,short_nan) 
%  Uses user-defined header struct
% [byteswritten,hdr] = cbiWriteNifti(filename,data,[],prec);
%  Creates a new header
% 
%  prec:      Precision (dataype) of output. Default (if no header is specified) is 'float32'. Use [] for default.
%             Should be a recognized Matlab (or nifti) data type string.
%             If precision of data is different than precision in header, data will be scaled
%             To avoid this, cast data to desired format before calling cbiWriteNifti, e.g.
%             cbiWriteNifti('myfilename',int16(data),hdr,'int16')
%             To avoid this, 
%  subset:    4x1 cell array describing image subset to save. 1-offset (Matlab-style).
%             Only the following options are supported:
%             - to save a single z-slice (e.g. 4):  subset={[],[],4,[]}
%             - to save a single/multiple volumes of a time series (e.g. volumes 2-9):  subset={[],[],[],[2 9]}
%  short_nan: NaN handling for signed short (int16) data. If 1, will treat save NaN's as -32768 (smallest 
%             representable number) reserving -32767..32767 for scaled data; otherwise will save NaN as 0.
%             Default is 1 (use -32768 as NaN).
% 
  if nargin < 2
    help cbiWriteNifti;
    return
  end
  
  if ~exist('subset', 'var')
    subset={[],[],[],[]};
  end
  if ~exist('short_nan', 'var')
    short_nan=1;
  end

  % 1D curvature file fix: 
  % check to see if this is 1D image that has a large second
  % dimension, if so, we need to reorder the data (this is 
  % necessary for surface data in which we *want* to store
  % a 1xnumVertices image with the surface coloring data,
  % but nifti's idiotic two byte width limitation prevents
  % us from doing so). Note that there is a complimentary 
  % piece of code in cbiReadNifti that handles unpacking
  % this data. Note that the realWidth is saved in the
  % header as part of the description (i.e. as 'realWidth:rest of description string')
  % -j.
  maxWidth = 2^15-1;realWidth = [];
  curvatureFix = false;
  if (size(data,1) == 1) && (size(data,2) > maxWidth)
    curvatureFix = true;
    realWidth = size(data,2);
    % find out how many rows we need to store the data
    numRows = ceil(realWidth/maxWidth);
    % fill the data out with nans past the real data
    data(1,end+1:numRows*maxWidth) = nan;
    % and reshape
    data = reshape(data,numRows,maxWidth);
    % print out what happened
    disp(sprintf(['(cbiWriteNifti) Saving a 1x%i image, reshaping to %ix%i to fit nifti limitation ' ...
                  'of max width of %i'], realWidth,numRows,maxWidth,maxWidth));
  end    
  
  if exist('cbihdr', 'var')
    hdr = niftifile(fname, cbiParseCBIHeader(cbihdr));
  else
    
    % If we don't have a pre-existing header, create empty one and try to fill relevant field
    % from data set.
    
    hdr = niftifile(fname);
    hdr.datatype = class(data);
    hdr = setDims(hdr, size(data));
  end
  
  if exist('prec', 'var') && ~isempty(prec)
    hdr.datatype = prec;
  end

  if curvatureFix
    % write the real width into the description
    % clip the length of the description to the max of 80
    descrip = sprintf('%i:%s',realWidth,hdr.descrip);
    hdr.descrip = descrip(1:min(end,80));
    % set correct dimensions
    hdr = setDims(hdr, size(data));
  end
  
  hdr = fopen(hdr, 'write');
  if ~exist('subset', 'var')
    subset={[1 hdr.nx],[1 hdr.ny],[1 hdr.nz],[1 hdr.nt]};
  end

  if ~strcmp(class(data), hdr.datatype)
    disp(['(cbiWriteNifti) Scaling data from ' class(data) ' to ' hdr.datatype]);
  end

  % get hdr scaling factor and convert data if necessary
  [data,hdr] = convertData(data,hdr,short_nan);  
  
  [hdr, pixelswritten] = fwrite(hdr, data, subset{1}, subset{2}, subset{3}, subset{4});
  hdr = fclose(hdr);
  if ~exist('cbihdr', 'var')
    cbihdr = cbiParseNiftiHeader(hdr);
  end
  byteswritten = pixelswritten * hdr.nbyper;
  
return
  
function [data,hdr] = convertData(data,hdr,short_nan);
% Scales and shifts data (using hdr.scl_slope and hdr.scl_inter)
% and changes NaN's to 0 or MAXINT for non-floating point formats
% Returns hdr with scale factor changed (bug fixed 20060824)

  matlabType = hdr.datatype;
  
  % Calculate scale factor for non-floating point data
  switch (matlabType)    
   case 'binary'
    error('unsupported format')
   case 'uint8'
    MAXINT=2^8-1;
   case {'uint16','ushort'}
    MAXINT=2^16-1;
   case {'uint32','uint'}
    MAXINT=2^32-1;
   case 'uint64'
    MAXINT=2^64-1;
   case 'int8'
    MAXINT=2^7-1;
   case {'int16','short'}
    MAXINT=2^15-1;
   case {'int32','int'}
    MAXINT=2^31-1;
   case 'int64'
    MAXINT=2^63-1;
   otherwise
    MAXINT=0;
  end
  if (MAXINT)
    hdr.scl_slope = max(data(:))/MAXINT;
  end
  
  % Scale and shift data if scale factor is nonzero
  if ~isnan(hdr.scl_slope) & (hdr.scl_slope ~= 0)
    if (hdr.scl_slope ~= 1) | (hdr.scl_inter ~= 0)
      data = double(data);
      data = (data - hdr.scl_inter) ./ hdr.scl_slope;
    end
  end
  
  % Change NaNs for non-floating point datatypes
  switch matlabType
   case {'binary','uint8','uint16','ushort','uint32','uint','int','uint64','int8','int16','int32','int64'}
    data(isnan(data)) = 0;
   case {'int16','short'}
    if (short_nan)
      data(isnan(data))=-32768;
    else
      data(isnan(data))=0;
    end
  end
  
  if ~strcmp(class(data), matlabType)
    switch matlabType
     case 'binary'
      data = binary(round(data));
     case 'uint8'
      data = uint8(round(data));
     case 'uint16'
      data = uint16(round(data));
     case 'short'
      data = short(round(data));
     case 'ushort'
      data = ushort(round(data));
     case 'uint32'
      data = uint32(round(data));
     case 'uint'
      data = uint(round(data));
     case 'int'
      data = int(round(data));
     case 'uint64'
      data = uint64(round(data));
     case 'int8'
      data = int8(round(data));
     case 'int16'
      data = int16(round(data));
     case 'int32'
      data = int32(round(data));
     case 'int64'
      data = int64(round(data));
     otherwise
      % nothing
    end
  end
    
function hdr = setDims(hdr, datadims)
  
  % loop through data dimensions and set values
    
  for dim = 1:7
    if dim <= length(datadims)
      hdr.dim(dim + 1) = datadims(dim);
    else
      if hdr.dim(dim + 1) ~= 1
        hdr.dim(dim + 1) = 1;
      end
    end
  end

