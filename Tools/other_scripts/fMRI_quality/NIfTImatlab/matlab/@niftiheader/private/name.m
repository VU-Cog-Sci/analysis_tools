function [fname, iname, type] = name(newname, type)
% [fname, iname, type] = name(newname, type)
%
% Store name 
%
% Parameters: newname - name to store
%             type    - current value
%
% Return: new values to be stored
%
% Notes: If name has extension, it takes precendence over type
  
  [basename, deftype, compressed] = findbase(newname);
  
  if (deftype ~= -1) && (deftype ~= type)
    if (deftype ~= 2) || ((type ~= 0) && (type ~= 2))
      type = int32(deftype);
    end
  end
  
  switch type
   case {0, 2}
    fname = strcat(basename, '.hdr');
    iname = strcat(basename, '.img');
   case 1
    fname = strcat(basename, '.nii');
    iname = strcat(basename, '.nii');
   case 3
    fname = strcat(basename, '.nia');
    iname = strcat(basename, '.nia');
  end
  
  if compressed
    fname = strcat(fname, '.gz');
    iname = strcat(iname, '.gz');
  end
  
  % Utility function to get name sans extension
  
function [base, deftype, compressed] = findbase(name)

  compressed = false;
  name = deblank(name);
  [path, base, ext] = fileparts(name);
  if strcmp(ext, '.gz')
    [path, base, ext] = fileparts(base);
    compressed = true;
  end

  % Rebuild the name without the extension
  
  base = fullfile(path, base);
  
  switch ext
   case '.hdr'
    deftype = 2;
   case '.img'
    deftype = 2;
   case '.nii'
    deftype = 1;
   case '.ima'
    deftype = 3;
   otherwise
    deftype = -1;
    base = name;
  end    
