function [nfd] = subsasgn(nfd, index, val)
% [nfd] = subsasgn(nfd, index, val)
%
% subsasgn 
%
  switch index(1).type
   case '()'
    error('Integer subscript indexing not supported by niftifile objects')
   case '{}'
    error('Cell array indexing not supported by niftifile objects')
   case '.'
    try
      nfd = set(nfd, index.subs, val);
    catch
      sprintf('%s', lasterr)
    end
  end
  
