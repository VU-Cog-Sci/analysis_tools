function nhdr = subsasgn(nhdr, index, val)
% nhdr = subsasgn(nhdr, index, val)
%
% Define index assignment for niftiheader objects
  switch index.type
   case '()'
    error('Integer subscript indexing not supported by niftiheader objects')
   case '.'
    try
      nhdr = set(nhdr, index.subs, val);
    catch
      sprintf('%s', lasterr)
    end
  end    
  
