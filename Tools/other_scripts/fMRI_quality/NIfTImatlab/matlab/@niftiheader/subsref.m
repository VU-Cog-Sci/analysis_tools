function val = subsref(nhdr, index)
% val = subsref(nhdr, index)
%
% Define field name indexing for niftiheader objects

  if size(index, 2) > 1
    error('niftiheader object cannot handle construct');
  else
    switch index.type
     case '()'
      error('Integer subscript indexing not supported by niftiheader objects');
     case '{}'
      error('Cell array indexing not supported by niftiheader objects');
     case '.'
      try 
        val = get(nhdr, index.subs);
      catch
      sprintf('%s', lasterr)
      end
    end
  end
  
