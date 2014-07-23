function nfd = cbiSetNiftiSform( nfd, mtx44, trns )
% nfd = cbiSetNiftiSform( nfd, mtx44 )
%  where mtx44 is a 4x4 homogeneous matrix
% - OR -
% nfd = cbiSetNiftiSform( nfd, R, T )
%  where R is a 3x3 rotation matrix and T a translation 3-vector
% 
% Sets sform44 and sform_code = 1

if nargin < 2
  error('Must specify a header and a qform matrix (or a rotation and translation matrix')
end
  
if nargin == 3
  m = eye(4);
  if size(mtx44) ~=[3 3]
    error ('rotation matrix must be 3x3');
  end
  m(1:3,1:3) = mtx44;
  if length(trns) ~= 3
    error('translation vector must be a 3-vector');
  end
  m(1:3,4) = trns(:);
  mtx44 = m;
end

nfd.niftiheader.sto_xyz = mtx44;

