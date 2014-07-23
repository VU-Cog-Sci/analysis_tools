function [fname, iname, nifti_type] = type(name, newtype)
% [fname, iname, nifti_type] = type(name, newtype)
%
% Store type
%
% Parameters: newtype - if specified, store new name in structure
%
% Return: hdr object
%
% Notes: 

  if nargin == 0
    fprintf(1, '\tnifti_type      : [ single | dual | analyze | ascii ]\n');
  else
    [basename, compressed] = findbase(name);
    if isnumeric(newtype)
      switch newtype
       case {2, 0}
        fname = strcat(basename, '.hdr');
        iname = strcat(basename, '.img');
       case 1
        fname = strcat(basename, '.nii');
        iname = strcat(basename, '.nii');
       case 3
        fname = strcat(basename, '.nia');
        iname = strcat(basename, '.nia');
      end
      nifti_type = int32(newtype);
    else
      switch lower(newtype)
       case {'dual', 'analyze'}
        fname = strcat(basename, '.hdr');
        iname = strcat(basename, '.img');
        if strcmp(lower(newtype), 'dual') == 1
          nifti_type = int32(2);
        else
          nifti_type = int32(0);
        end
       case 'single'
        fname = strcat(basename, '.nii');
        iname = strcat(basename, '.nii');
        nifti_type = int32(1);
       case 'ascii'
        fname = strcat(basename, '.nia');
        iname = strcat(basename, '.nia');
        nifti_type = int32(3);
      end
    end
  end
  
  if compressed
    fname = strcat(fname, '.gz');
    iname = strcat(iname, '.gz');
  end
  
  % Utility function to get name sans extension
  
function [base, compressed] = findbase(name)
  
  compressed = false;
  name = deblank(name);
  [path, base, ext] = fileparts(name);
  if strcmp(ext, '.gz')
    [path, base, ext] = fileparts(base);
    compressed = true;
  end
  
  if ~strcmp(ext, '.hdr') && ~strcmp(ext, '.nii') && strcmp(ext, '.ima')
    base = name;
  else
    base = fullfile(path, base);
  end    
