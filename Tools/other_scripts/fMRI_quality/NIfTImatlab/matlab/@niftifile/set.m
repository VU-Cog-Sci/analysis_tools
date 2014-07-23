function nfd = set(nfd, varargin)
% nfd = set(nfd, propname, propval, ...)
%
% Set poperty values for niftifile object
%

  if ~isa(nfd,'niftifile'),
   % Call built-in SET. 
   builtin('set', nfd, varargin{:});
   return
  end

  if length(varargin) == 0
    if nargout > 0
      nfd.niftiheader = set(nfd.niftiheader);
    else
      set(nfd.niftiheader);
    end
  elseif length(varargin) == 1
    if nargout > 0
      nfd.niftiheader = set(nfd.niftiheader, varargin);
    else
      set(nfd.niftiheader, varargin);
    end
  else
    propertyArgIn = varargin;
    while length(propertyArgIn) >= 2
      prop = propertyArgIn{1};
      val = propertyArgIn{2};
      propertyArgIn = propertyArgIn(3:end);
      try
        switch prop
         case 'name'
          nfd = fclose(nfd);
          nfd.niftiheader.fname = val;
          nfd.compressed = (length(val) > 3) && strcmp(val(length(val)-2:end), '.gz');          
         case {'type', 'nifti_type'}
          nfd = fclose(nfd);
          nfd.niftiheader.nifti_type = val;
         case 'compressed'
          nfd = fclose(nfd);
          nfd.compressed = val;
          nfd.niftiheader.compressed = val;
         case 'filePos'
          error([prop,' Is not a settable niftiheader field'])
         case 'fileId'
          error([prop,' Is not a settable niftiheader field'])
         otherwise
          nfd.niftiheader = set(nfd.niftiheader, prop, val, propertyArgIn);
        end
      catch
        sprintf('%s', lasterr)
      end
    end      
  end
  
