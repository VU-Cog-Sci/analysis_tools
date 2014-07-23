function [quatern, qfac, pixdim] = mat44ToQuat(mat44)

%  
% Calculates quaternion parameters from a 4x4 matrix
% [quatern, qfac, pixdim] = mat442quat(mat44)
% 
% Taken from code written by Jonas Larsson which was based on c function 
%  nifti_mat44_to_quatern in nifti1_io.c (nifticlib)  
  
  % Use only 3x3 matrix
    
  mat33 = mat44(1:3, 1:3);
    
  %  compute lengths of each column; these determine grid spacings
 
  xd = sqrt( sum(mat33(:,1).^2) );
  yd = sqrt( sum(mat33(:,2).^2) );
  zd = sqrt( sum(mat33(:,3).^2) );
 
  
  % if a column length is zero, patch the trouble 
  
  if ( xd == 0 )
    mat33(1,1) = 1; 
    mat33(2,1) = 0; 
    mat33(3,1) = 0; 
    xd = 1;
  end
  if ( yd == 0 )
    mat33(2,2) = 1; 
    mat33(1,2) = 0; 
    mat33(3,2) = 0; 
    yd = 1; 
  end
  if ( zd == 0 )
    mat33(3,3) = 1; 
    mat33(1,3) = 0; 
    mat33(2,3) = 0; 
    zd = 1; 
  end
  pixdim = [xd yd zd];

  % normalize the columns
    
  mat33(:,1) = mat33(:,1)./ xd;
  mat33(:,2) = mat33(:,2)./ yd;
  mat33(:,3) = mat33(:,3)./ zd;
  
  mat33 = polarDecomp33(mat33);
  zd = det(mat33);
  
  if (zd > 0)
    qfac = 1;
  else
    qfac = -1;
    mat33(:,3) = -mat33(:,3);
  end
  
  a = mat33(1,1) + mat33(2,2) + mat33(3,3) + 1;
  if (a > 0.5 )
    a = 0.5 * sqrt(a) ;
    b = 0.25 * (mat33(3,2) - mat33(2,3)) / a ;
    c = 0.25 * (mat33(1,3) - mat33(3,1)) / a ;
    d = 0.25 * (mat33(2,1) - mat33(1,2)) / a ;
  else
    xd = 1.0 + mat33(1,1) - (mat33(2,2) + mat33(3,3)) ;
    yd = 1.0 + mat33(2,2) - (mat33(1,1) + mat33(3,3)) ;
    zd = 1.0 + mat33(3,3) - (mat33(1,1) + mat33(2,2)) ;
    if( xd > 1.0 )
      b = 0.5 * sqrt(xd) ;
      c = 0.25 * (mat33(1,2) + mat33(2,1)) / b ;
      d = 0.25 * (mat33(1,3) + mat33(3,1)) / b ;
      a = 0.25 * (mat33(3,2) - mat33(2,3)) / b ;
    elseif ( yd > 1.0 )
      c = 0.5 * sqrt(yd) ;
      b = 0.25 * (mat33(1,2) + mat33(2,1)) / c ;
      d = 0.25 * (mat33(2,3) + mat33(3,2)) / c ;
      a = 0.25 * (mat33(1,3) - mat33(3,1)) / c ;
    else
      d = 0.5 * sqrt(zd) ;
      b = 0.25 * (mat33(1,3) + mat33(3,1)) / d ;
      c = 0.25 * (mat33(2,3) + mat33(3,2)) / d ;
      a = 0.25 * (mat33(2,1) - mat33(1,2)) / d ;
    end
    if( a < 0.0 )
      b = -b; 
      c = -c; 
      d = -d; 
      a = -a; 
    end
    
  end
  quatern = [b c d];

function OM = polarDecomp33(M);
  
   X = M;

   gam = det(X) ;
   while ( gam == 0.0 )
     gam = 0.00001 * ( 0.001 + norm(X,'inf'));
     X(1,1) = X(1,1) + gam;
     X(2,2) = X(2,2) + gam;
     X(3,3) = X(3,3) + gam;
     gam = det(X);
   end
   dif = 1;
   k = 0;
   while (1)
     Y = inv(X);
     if( dif > 0.3 )
       alp = sqrt( norm(X,'inf') * norm(X','inf'));
       bet = sqrt( norm(Y,'inf') * norm(Y','inf'));
       gam = sqrt( bet / alp ) ;
       gmi = 1.0 / gam ;
     else
       gam = 1.0;
       gmi = 1.0 ;
     end

     Z(1,1) = 0.5 * ( gam * X(1,1) + gmi * Y(1,1) ) ;     
     Z(1,2) = 0.5 * ( gam * X(1,2) + gmi * Y(2,1) ) ;
     Z(1,3) = 0.5 * ( gam * X(1,3) + gmi * Y(3,1) ) ;
     Z(2,1) = 0.5 * ( gam * X(2,1) + gmi * Y(1,2) ) ;
     Z(2,2) = 0.5 * ( gam * X(2,2) + gmi * Y(2,2) ) ;
     Z(2,3) = 0.5 * ( gam * X(2,3) + gmi * Y(3,2) ) ;
     Z(3,1) = 0.5 * ( gam * X(3,1) + gmi * Y(1,3) ) ;
     Z(3,2) = 0.5 * ( gam * X(3,2) + gmi * Y(2,3) ) ;
     Z(3,3) = 0.5 * ( gam * X(3,3) + gmi * Y(3,3) ) ;

     dif = abs(Z(1,1) - X(1,1)) + abs(Z(1,2) - X(1,2)) ...
	   + abs(Z(1,3) - X(1,3)) + abs(Z(2,1) - X(2,1)) ...
	   + abs(Z(2,2) - X(2,2)) + abs(Z(2,3) - X(2,3)) ...	   
	   + abs(Z(3,1) - X(3,1)) + abs(Z(3,2) - X(3,2)) ...
	   + abs(Z(3,3) - X(3,3));

     k = k + 1 ;
     if ( k > 100 | dif < 3.e-6 ) 
       break ;
     end
     X = Z ;
   end

   OM = Z;
   return
