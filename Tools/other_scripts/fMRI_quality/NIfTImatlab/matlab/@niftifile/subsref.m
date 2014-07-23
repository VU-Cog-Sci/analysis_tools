function [val] = subsref(nfd, index)
% [val] = subsref(nfd, index)
%
% Define field name indexing for niftifile objects
%  
  switch index(1).type
%%%   case '()'
%%%    error('Integer subscript indexing not supported by niftifile objects')
%%%   case '{}'
%%%    error('Cell array indexing not supported by niftifile objects')
   case '.'
    try
      val = get(nfd, index.subs);
    catch
      val = 0;
      fprintf(1, '%s\n', lasterr)
    end
  end
  
