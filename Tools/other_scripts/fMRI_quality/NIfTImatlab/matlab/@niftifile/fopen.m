function [nfd] = fopen(nfd, mode)
% [nfd] = fopen(nfd, mode)
%
% Open the file associated with this object and process header (write or read)
%
% Parameters: nfd  - the niftifile object
%             mode - 'read', 'write', 'update'
%
% Returns: nfd - the modified niftifile object
%
% Notes: Before using the function, the niftifile object needs to be created using the
%        constructor. 
%        Example:
%                     nfd = niftifile('image.nii');
%        
%        For more information on the class and constructor: 'help niftifile'  
%        
%        In "read" mode, the function will open the file and read the header. To read the data use
%        'fread()' ('help niftifile/fread')
%        
%        In "write" mode, the function will create the file and write the header that was created
%        in memory. The header must be properly filled with the necessary values, some basic
%        checking for a properly formed header is performed, but it will not be able to detect if
%        some fields do not have the right values. To write data use 'fwrite()' ('help
%        niftifile/write').
%
%        In "update" mode, if this is a new file it will behave as "write", else allows the user to
%        update part of the file without modifying the rest.
  
  
  % Basic error checking
    
  if nargin ~= 2
    error('Usage: nfd = fopen(nfd, mode);');
  end
  
  if ~isa(nfd, 'niftifile');
    error('First input parameter must be niftifile object');
  end

  % Check if it is already open

  if nfd.fileID ~= 0
    error('File already open, use fclose(nfd) first');
  end

  % Use the system 'echo' command to expand the name correctly
  
  [status, result] = system(['echo ', nfd.niftiheader.fname]);
  if status == 0
    nfd.niftiheader.fname = deblank(result);
  else
    error(['Illegale file name ', nfd.niftiheader.fname]);
  end
  
  switch mode
   case 'read'
    
   if (exist(nfd.niftiheader.fname, 'file')) == 0
     error(['No such file: ', nfd.niftiheader.fname]);
   else
     % Test open to avoid nasty crash dump
     [fid, message] = fopen(nfd.niftiheader.fname, 'r');
     if fid < 0
       error([message, ': ', nfd.niftiheader.fname]);
     else
       fclose(fid);
     end
   end
    % Use c function to open file and read header
  
    [nfd.fileID, nfd.niftiheader] = niftiMatlabIO('OPEN', nfd.niftiheader);

   case 'write'
    
    % Verify that header is good
  
    if niftiMatlabIO('VERIFY', nfd.niftiheader) ~= 1
      error('Malformed header');
    end

    % Just try to open for write and let the system tell you if it's OK
    
    [fid, mess] = fopen(nfd.niftiheader.fname, 'w');
    if fid < 0
      error([mess, ': ', nfd.niftiheader.fname]);
    else
      fclose(fid);
    end

    [nfd.fileID, nfd.niftiheader] = niftiMatlabIO('CREATE', nfd.niftiheader);
    
   case 'update'
    
    % If file does not exist it's really a 'write', else open for update

    status = exist(nfd.niftiheader.fname, 'file');
    if status == 0
      nfd = fopen(nfd, 'write');
    else

      % Verify that header is good
  
      if niftiMatlabIO('VERIFY', nfd.niftiheader) ~= 1
        error('Malformed header');
      end

      % Just try to open for write and let the system tell you if it's OK
    
      [fid, mess] = fopen(nfd.niftiheader.fname, 'a');
      if fid < 0
        error([mess, ': ', nfd.niftiheader.fname]);
      else
        fclose(fid);
      end

      [nfd.fileID, nfd.niftiheader] = niftiMatlabIO('UPDATE', nfd.niftiheader);
    
    end
   otherwise
    error('What you talking about !?')
    
  end

  if nfd.fileID > 0
    nfd.hdrProcessed = true;
  end

  nfd = fseek(nfd, 0, 'bof');
  name = nfd.niftiheader.fname;
  nfd.compressed = (length(name) > 3) && strcmp(name(length(name)-2:end), '.gz');
  nfd.mode = mode;

