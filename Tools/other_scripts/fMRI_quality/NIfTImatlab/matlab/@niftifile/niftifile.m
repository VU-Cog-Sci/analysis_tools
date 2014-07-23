function [nfd] = niftifile(name, iniOb)
% [nfd] = niftifile(name, iniOb)
%
% niftifile Constructor
%
% Create a niftifile object, derived from niftiheader and with file information
%
%   nfd = niftifile()		Create object with default values for niftiheader
%                       	File name will have to be set with 'name' assignment.
%
%   nfd = niftifile(NAME)	Create object with default values and set file to NAME
%				If extension is present, nifti type will be inferred with
%				precendence given to NIfTI dual file over ANALYZE 7.5.
%				If no extension given, type will have to be set with 'type'
%				assignment. 
% 
%   nfd = niftifile(INIOB)      Create object using values from INIOB object.
%
%   nfd = niftifile(NAME,INIOB)	Create object using values from INIOB object, set file to NAME. 
% 				If name contains extension which is not compatible with type in
% 				INIOB, new type will be inferred with precendence given to NIfTI
% 				dual file over ANALYZE 7.5. 
%
% Return: nifti file handle to be used for all I/O ops
%
%
% Special assignments:
%
%   nfd.name = NAME		Store name using type and compression information
%   nfd.type = TYPE		Modify file names to be consistent with TYPE
%				TYPE = 'ANALYZE', 'SINGLE', 'DUAL', 'ASCII'
%   nfd.compress = true/false	Modify file names to be consistent with value
%
% Methods:
%
%   fopen(nfd, mode)			= open file for 'read' or 'write', process header
%   fclose(nfd)                         
%   fread(nfd, <num pixels>)		= read pixels from file
%   fwrite(nfd, data, <num pixels>)     = write pixels to file
%   fseek(nfd, offset, [origin])  	= set file position to offset bytes from origin
  
  % Basic error checking
    
  if nargin > 2
    error('Easy cowboy, I cannot take more than 2 arguments !');
  end
  
  if nargin == 1
    if isa(name, 'niftifile')
      iniOb = name;
    else
      if ~ischar(name)
        error('First argument must be a string or niftifile object');
      end
    end
  end
 
  if (nargin > 1) && (~isa(iniOb, 'niftifile'))
      error('INIOB must be a niftifile object');
  end
  
  if nargout > 1
    error('What am i supposed to do with all those output arguments ?');
  end

  % Object members
  %     mode            = open mode, used for subsequent sanity checks
  %     filePos		= where we are in the file, excluding the header
  %     fileId          = id of the file, if open
  %     hdrProcessed	= flag to indicate if header was read/written
  %     compressed	= flag to indicate compression
  
  % Store default values
  
  nfd.mode = '';
  nfd.hdrProcessed = false;
  nfd.compressed = false;
  nfd.fileID = int32(0);
  nfd.filePos = int32(0);
  nfd.header_type = 'nifti_image';
    
  if nargin == 0
    nhdr = niftiheader;
  else
    if nargin == 1
      if isa(name, 'niftifile')
        nhdr = iniOb.niftiheader;
      else
        nhdr = niftiheader(name);
      end
    else 
      nhdr = iniOb.niftiheader;
      nhdr.name = name;
    end
  end

  if exist('name', 'var')
    nfd.compressed = (length(name) > 3) && strcmp(name(length(name)-2:end), '.gz');
  end
  
  % The class is derived from niftiheader
  
  nfd = class(nfd, 'niftifile', nhdr);
