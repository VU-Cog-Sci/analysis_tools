function [mat44] = quatToMat44(nhdr)
  
%
% Taken from nifti1_io.c (nifticlib) code
% code originally written by Jonas Larsson
%
  qb = nhdr.quatern_b;
  qb2 = qb * qb;
  qc = nhdr.quatern_c;
  qc2 = qc * qc;
  qd = nhdr.quatern_d;
  qd2 = qd * qd;
  qa2 = 1 - (qb2 + qc2 + qd2);

  if ( qa2 < 1.0e-7 )                   
    % Special case
    % qa = 0 ==> 180 degree rotation    
    qa = 0;
    qa2 = 0;
    % Normalize (qb, qc, qd) vector    
    qb = qb / sqrt(qb2 + qc2 + qd2);
    qc = qc / sqrt(qb2 + qc2 + qd2);
    qd = qd / sqrt(qb2 + qc2 + qd2);
  else
    qa = sqrt(qa2);
  end
  
  R = [ (qa2 + qb2 - qc2 - qd2)    (2 * (qb * qc - qa * qd))  (2 * (qb * qd + qa * qc)); ...
	(2 * (qb * qc + qa * qd))  (qa2 + qc2 - qb2 - qd2)    (2 * (qc * qd - qa * qb)); ...
	(2 * (qb * qd - qa * qc))  (2 * (qc * qd + qa * qb))  (qa2 + qd2 - qc2 - qb2) ];

  if (nhdr.qfac >= 0)
    qfac = 1;
  else
    qfac = -1;
  end
  
  mat44 = eye(4);
  mat44(1:3,1:3) = (R * diag([ nhdr.pixdim(2) nhdr.pixdim(3) (qfac * nhdr.pixdim(4)) ]));
  mat44(1:3,4) = [ nhdr.qoffset_x nhdr.qoffset_y nhdr.qoffset_z ];
