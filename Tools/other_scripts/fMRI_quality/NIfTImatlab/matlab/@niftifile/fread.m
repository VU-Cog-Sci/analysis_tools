function [nfd, data, size] = fread(nfd, varargin)
% [nfd, data, size] = fread(nfd, numpix)
%
% Read pixels from file  
%
% Parameters: nfd	- nifti header
%             varargin  - either 1 value = number of pixels to read
%                         or     list of ranges for each dimension
%
% Return:     nfd       - modified nifti header
%             data      - the data
%             size	- number of pixels read
%
% Notes: If ranges are specified, at least 2 are needed.
%        Unspecified outer ranges default to "all"
%        If ranges are specified, reading allways begins at the biginning of data
%  
% Examples:
%    [nfd, data, size] = fread(nfd, numpix);
%    [nfd, data, size] = fread(nfd, [1], [28 36], [1 16]);
%        if third dimension is 16, the above can be called as:
%    [nfd, data, size] = fread(nfd, [1], [28 36]);
  
  % Make sure this is set up for reading
  
  if nargin < 1
    error('usage: [nfd, data, size] = fred(nfd, varargin)');
  end
  
  if (nfd.fileID == 0) || strcmp(nfd.mode, 'read') == 0
    error('File not opened for read');
  end
  
  ranges = nargin > 2;
  numpix = int32(1);
  
  if ranges 
    numpix = checkranges(nfd, varargin);
  else
    if nargin == 1
      numpix = int32(nfd.nvox);
    else
      if ~isnumeric(varargin{1})
        error('Number of pixels must be integer type');
      end
      numpix = int32(varargin{1});
    end
  end
  
  bytesPer = nfd.niftiheader.nbyper;
  numBytes = numpix * bytesPer;
  
  % If data is complex, unpack it
  if (strcmp(nfd.niftiheader.datatype, 'complex'))
    if ranges
      [size, comBuffer, nfd.filePos, nfd.niftiheader] = niftiMatlabIO('READCHUNK', nfd.fileID, ...
                                                        nfd.niftiheader, varargin);  
    else
      [size, comBuffer, nfd.filePos, nfd.niftiheader] = niftiMatlabIO('READ', nfd.fileID, numBytes, nfd.niftiheader);
    end
    data = comBuffer(1:2:2*numpix-1) + i * comBuffer(2:2:2*numpix);
  else
    if ranges
      [size, data, nfd.filePos, nfd.niftiheader] = niftiMatlabIO('READCHUNK', nfd.fileID, ...
                                                        nfd.niftiheader, varargin); 
    else
      [size, data, nfd.filePos, nfd.niftiheader] = niftiMatlabIO('READ', nfd.fileID, numBytes, nfd.niftiheader);
    end
  end
  
  size = size / bytesPer;

  % Compute pixels from bytesa
  
  if nfd.niftiheader.nifti_type == 1
    nfd.filePos = nfd.filePos - nfd.niftiheader.iname_offset;
  end
  nfd.filePos = nfd.filePos / bytesPer;
  