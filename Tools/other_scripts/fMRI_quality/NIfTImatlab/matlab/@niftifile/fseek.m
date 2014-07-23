function [nfd] = fseek(nfd, offset, origin)
% [nfd] = fseek(nfd, offset, origin)
%
% Do an fseek in the file
%
% Parameters: nfd	- current object
%	      offset    - how many pixels to skip
%             origin    - optional starting point: 
%			     SET | bof | -1 - beginning of file - *DEFAULT*
%  			     CUR | cof |  0 - current position
%			     END | eof |  1 - end of file - *Implies negative movement*
%
% Notes: This is movement within the data portion of file, i.e., if it's a NIfTI single file skip
%        the header.
%        The user indicates how many pixels to skip.
  
  if ~isnumeric(offset)
    error('Number of pixel offset must be integer type');
  end
  
  if ~isinteger(offset)
    offset = int32(offset);
  end

  % Set to default of BOF
  
  whence = 0;
  
  if nargin == 3
    if ischar(origin)
      switch lower(origin)
       case {'cur', 'cof'}
        whence = 1;
       case {'end', 'eof'}
        whence = 2;
       case {'set', 'bof'}
        whence = 0;
       otherwise
        error(['Invalid origin value ', origin]);
      end
    else
      switch origin
       case {-1, 0, 1}
        whence = origin + 1;
       otherwise
        error(['Invalid origin value ', origin]);
      end
    end
  end

  % Make sure we don't "break" the file
  
  if (offset + nfd.filePos < 0)
    error('Baking up past beginning of data');
  end
  
  % Compute bytes from pixels
  
  offset = offset * nfd.niftiheader.nbyper;

  % If this is single file, skip header
  
  if (nfd.niftiheader.nifti_type == 1) & (whence == 0)
      offset = offset + nfd.niftiheader.iname_offset;
  end
  
  nfd.filePos = niftiMatlabIO('SEEK', offset, whence, nfd.fileID);

  % Subtract header and compute pixels from bytes
  
  if nfd.niftiheader.nifti_type == 1
    nfd.filePos = nfd.filePos - nfd.niftiheader.iname_offset;
  end
  nfd.filePos = nfd.filePos / nfd.niftiheader.nbyper;