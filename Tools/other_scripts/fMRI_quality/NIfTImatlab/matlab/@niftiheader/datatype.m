function nhdr = datatype(nhdr, val)
  
%  nhdr = datatype(nhdr, value)
%
%  Change the value of the datatype and associated bytes-per-pixel and swapsize
  
  if nargin < 2
      fprintf(1, '  datatype : [ uint8 | uint16 | ushort | uint32 | uint | uint64 | int8');
      fprintf(1, ' | int16 | short | int32 | int | int64 | single | float | double | float128');
      fprintf(1, ' | RGB | complex | complex128 | complex256 ]\n');
  else
    % Use symbolic names for data type and set bytes per pix
    switch lower(val)
     case 'uint8'
      nhdr.datatype = int32(2);
      nhdr.nbyper = int32(1);
      nhdr.swapsize = int32(0);
     case {'int16', 'short'}
      nhdr.datatype = int32(4);
      nhdr.nbyper = int32(2);
      nhdr.swapsize = int32(2);
     case {'int32', 'int'}
      nhdr.datatype = int32(8);
      nhdr.nbyper = int32(4);
      nhdr.swapsize = int32(4);
     case {'single', 'float'}
      nhdr.datatype = int32(16);
      nhdr.nbyper = int32(4);
      nhdr.swapsize = int32(4);
     case 'complex'
      nhdr.datatype = int32(32);
      nhdr.nbyper = int32(8);
      nhdr.swapsize = int32(4);
     case 'double'
      nhdr.datatype = int32(64);
      nhdr.nbyper = int32(8);
      nhdr.swapsize = int32(8);
     case 'RGB'
      nhdr.datatype = int32(128);
      nhdr.nbyper = int32(3);
      nhdr.swapsize = int32(0);
     case 'int8'
      nhdr.datatype = int32(256);
      nhdr.nbyper = int32(1);
      nhdr.swapsize = int32(0);
     case {'uint16', 'ushort'}
      nhdr.datatype = int32(512);
      nhdr.nbyper = int32(2);
      nhdr.swapsize = int32(2);
     case {'uint32', 'uint'}
      nhdr.datatype = int32(768);
      nhdr.nbyper = int32(4);
      nhdr.swapsize = int32(4);
     case 'int64'
      nhdr.datatype = int32(1024);
      nhdr.nbyper = int32(8);
      nhdr.swapsize = int32(8);
     case 'uint64'
      nhdr.datatype = int32(1280);
      nhdr.nbyper = int32(8);
      nhdr.swapsize = int32(8);
     case 'float128'
      nhdr.datatype = int32(1536);
      nhdr.nbyper = int32(16);
      nhdr.swapsize = int32(16);
     case 'complex128'
      nhdr.datatype = int32(1792);
      nhdr.nbyper = int32(16);
      nhdr.swapsize = int32(8);
     case 'complex256'
      nhdr.datatype = int32(2048);
      nhdr.nbyper = int32(32);
      nhdr.swapsize = int32(16);      
     otherwise
      error([val,' is not one of valid data types: uint8, uint16, ushort, uint32, uint, ' ...
             'uint64, int8, int16, short, int32, int, int64, single, float, double, float128, ' ...
             'RGB, complex, complex128, complex256']);
    end

  end
  
