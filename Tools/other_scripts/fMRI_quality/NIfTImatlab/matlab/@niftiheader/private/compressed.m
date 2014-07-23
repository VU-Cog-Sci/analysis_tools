function [fname, iname] = compressed(fname, iname)
% [fname, iname] = compressed(fname, iname)
%
% Modify file names to reflect compression
%
% Parameters: None
%
% Return: None
%
% Notes: 

  if nargin == 0
      fprintf(1, '\tcompressed      : [ true | false ]\n');
  else
    fname = strcat(fname, '.gz');
    iname = strcat(iname, '.gz');
  end
