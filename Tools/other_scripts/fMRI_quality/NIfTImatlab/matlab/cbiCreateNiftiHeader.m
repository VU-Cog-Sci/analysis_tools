function cbihdr = cbiCreateNiftiHeader(varargin)
% cbihdr = cbiCreateNiftiHeader(varargin)
%
% Generates a NIFTI-compatible Matlab header struct that can be saved with cbiWriteNiftiHeader.
% 
% SYNTAX:
% cbihdr = cbiCreateNiftiHeader
%     No input arguments: generates a bare-bone header struct with correctly sized fields. 
%     Sets endianness and default data type (float), with voxel size = 1 mm isotropic.     
% cbihdr = cbiCreateNiftiHeader(data)
%     Input is a data array: generates a header to match data (essentially as option (1) but also sets hdr.dims)
% cbihdr = cbiCreateNiftiHeader(hdr)
%     Input is a header struct: adds missing fields to an existing header and ensures consistency.
% cbihdr = cbiCreateNiftiHeader(hdr_field1,value1,hdr_field2,value2,...)
%     Input is a set of parameter-value pairs, e.g. ( 'srow_x', [0 0 0 1] ). Sets the corresponding field.
%
%     The output is checked to make sure it is consistent (within limits). Specifically:
%     - If a data array is given as input, hdr.dim will be set from this (overriding parameter-value pairs)
%     - Quaternions will be set as follows:
%       a) if neither qform44, or quatern_[b/c/d] are specified, will use existing hdr info 
%          (if valid, qform44 takes precedence over quaternions)
%       b) specifying a quaternion/qoffset as a parameter sets qform44 accordingly
%       c) specifying a qform44 sets quaternions accordingly (overriding any quaternion parameters)
%       d) if qform44 is empty, qform_code = 0, else qform_code = 1
%     - Sform data will be set as follows:
%     - a) if neither sform44 or srow_[x/y/z] are specified, will use  existing hdr info 
%          (if valid, sform44 takes precedence over srow_[x/y/z]
%     - b) specifying a srow_ paramater sets sform44 accordingly
%     - c) specifying sform44 sets srows accordingly (overriding any srow parameters)
%     - d) if sform44 is empty, sform_code = 0, else sform_code = 1
%     - Setting the file name also sets the magic number
% 
% Options 2-4 can be combined. The data and header input arguments must come before the parameter-value pairs.
  
  if nargin == 0
    % create empty header
    hdr = niftifile();
  end
  
  % Check if the input contains a data or existing header struct
  input_hdr = 0;
  input_data = 0;
  for currarg = 1:min([nargin;2])
    if isa(varargin{currarg}, 'niftifile')
      input_hdr = currarg;
    elseif ~isstr(varargin{currarg})
      if currarg == 1
	% must be a data array - parameter/value pairs always start with a string
	input_data = currarg;
      else
	% can only be a data array if the first argument was the header
	if input_hdr == 1
	  input_data = currarg;
	end
      end
    end
  end

  % Make a header
  if input_hdr
    hdr = niftifile(varargin{input_hdr});
  else
    hdr = niftifile();
  end

  % Set parameter-value pairs
  argstart = max([input_hdr input_data])+1;
  if argstart < nargin
    if rem(nargin-argstart+1, 2)
      error('Missing value for parameter-value pairs');
    end
    
    for currarg = argstart:2:nargin
      if ~isstr(varargin{currarg})
	error('Incorrect parameter specification (expected a field name)');
      end
      hdr = set(hdr, varargin{currarg}, varargin{currarg+1});

    end
  end
  
  % Set data size, if any
  if input_data
    s = size(varargin{input_data});    
    l = length(s);
    hdr.dim(2:l+1) = s;
    if l>2
      hdr.slice_end = s(3)-1; % dim3 (Z) is slice direction
    else
      hdr.slice_end = 0;
    end
    % Set pixdim fields, if not set
    for n = 2:l+1
      if (hdr.pixdim(n) == 0)
	hdr.pixdim(n) = 1;
      end
    end
  end
    
  % Now convert this to the struct header
  
  cbihdr = cbiParseNiftiHeader(hdr);
  
return
