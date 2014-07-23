function [cbihdr,fid] = cbiWriteNiftiHeader(cbihdr,fname,no_overwrite,leave_open)
% [hdr,fid] = cbiWriteNiftiHeader(hdr,fname [,no_overwrite,leave_open])
% 
% Saves a NIFTI-1 file header using the hdr struct provided.
% See cbiReadNiftiHeader for details of header structure.
%
% fname can be a file name or a file pointer.
% If fname is a file name: if the extension is:
%  - .hdr/.img: Will open or create the .hdr file and write to BOF.
%  - .nii: Will open or create a .nii file and write to BOF.
%  Unless the no_overwrite flag is nonzero, any existing file with the same name will
%  be deleted. (Use no_overwrite to replace the header data without destroying the
%  image data in a .nii file). This flag is ignored for dual file types (.img file is
%  not affected by this function).
%
% If fname is not specified or empty, will use the hdr.hdr_name field, if it exists.
% If this field is not set, fname must be specified.
%
% If leave_open is nonzero, will leave the file open for writing and return a 
% pointer to the open file (appropriate for writing image data to a .nii file)
% This flag is ignored for dual (hdr/img) file types.
% 

  if ~exist('leave_open', 'var')
    leave_open = 0;
  end
  if ~exist('fname', 'var') | isempty(fname)
    if ~isfield(cbihdr,'hdr_name') | isempty(cbihdr.hdr_name)
      error('No file name specified!');
    else
      fname = cbihdr.hdr_name;
    end
  end

  % If this was an open file, get the name, close it and start all over

  if ~isstr(fname)
    fid = fname;
    fname = fopen(fid);
    fclose(fid);
  end
  
  hdr = cbiParseCBIHeader(cbihdr);
  hdr = niftifile(fname, hdr);
  hdr = fopen(hdr, 'update');

  % Close file and, if needed, reopen it
  
  hdr = fclose(hdr);
  fid = [];
  
  if leave_open
    fid = fopen(hdr.fname, 'r+', cbihdr.endian);
  end

  return