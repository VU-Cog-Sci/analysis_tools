function [nfd, size] = fwrite(nfd, data, varargin)
% [nfd, size] = fwrite(nfd, data, numpix)
%
% Write pixels to file  
%
% Parameters: nfd	- nifti header
%             data      - what else 
%             varargin  - either 1 value = number of pixels to write
%                         or     list of ranges for each dimension
%
% Return:     nfd       - modified nifti header
%             size	- number of pixels written             
%
% Notes: If specifying ranges, at least 2 are needed.
%        Unspecified outer ranges default to "all"  
%  
% Examples:
%    [nfd, size] = fwrite(nfd);
%    [nfd, size] = fwrite(nfd, data, numpix);
%    [nfd, data, size] = fwrite(nfd, data, [1], [28 36], [1 16]);
%        if third dimension is 16, the above can be called as:
%    [nfd, data, size] = fwrite(nfd, data, [1], [28 36]);
  
% Start by verifying file is in write mode
  
  if (strcmp(nfd.mode, 'write') == 0) & (strcmp(nfd.mode, 'update') == 0)
    error('File not opened for write');
  end

  if (nargin < 3)
    error('Usage: [nfd, size] = fwrite(nfd, data, varargin)');
  end
  
  % Verify that header is reasonable
  
  if niftiMatlabIO('VERIFY', nfd.niftiheader) ~= 1
    error('Malformed header');
  end

  ranges = nargin > 3;
  numpix = int32(1);
    
  if ranges
    numpix = checkranges(nfd, varargin);
  else
    if ~isnumeric(varargin{1})
      error('Number of pixels must be integer type');
    end
      
    numpix = int32(varargin{1});
  end

  % If data on disk is complex, verify that data point is single
  
  if ( (strcmp(nfd.niftiheader.datatype, 'complex') && isreal(data)) ...
       || (~strcmp(nfd.niftiheader.datatype, 'complex') && ~ strcmp(class(data), nfd.niftiheader.datatype)) )
    error('Inconsistent data types');
  end
  
  bytesPer = nfd.niftiheader.nbyper;
  numBytes = numpix * bytesPer;
  
  % If data is complex, pack it
  
  if strcmp(nfd.niftiheader.datatype,'complex')
    compBuffer = zeros(2 * numpix, 1, 'single');
    compBuffer(1:2:numpix * 2) = single(real(data(1:numpix)));
    compBuffer(2:2:numpix * 2) = single(imag(data(1:numpix)));
    if ranges
      [size, nfd.filePos] = niftiMatlabIO('WRITECHUNK', nfd.fileID, nfd.niftiheader, compBuffer, varargin);
    else
      [size, nfd.filePos] = niftiMatlabIO('WRITE', nfd.fileID, nfd.niftiheader, compBuffer, numBytes); 
    end
  else
    if ranges
      [size, nfd.filePos] = niftiMatlabIO('WRITECHUNK', nfd.fileID, nfd.niftiheader, data, varargin);
    else
      [size, nfd.filePos] = niftiMatlabIO('WRITE', nfd.fileID, nfd.niftiheader, data, numBytes);
    end
  end
  
  % Compute pixels from bytes

  if nfd.niftiheader.nifti_type == 1
    nfd.filePos = nfd.filePos - nfd.niftiheader.iname_offset;
  end
  nfd.filePos = nfd.filePos / bytesPer;
  