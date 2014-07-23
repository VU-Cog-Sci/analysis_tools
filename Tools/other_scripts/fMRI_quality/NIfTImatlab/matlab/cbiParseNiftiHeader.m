function cbihdr = cbiParseNiftiHeader(niftihdr)

% Take a niftimatlab header (niftihdr) and generate an mrUtils header (cbihdr)

  cbihdr.single_file = 0;
  cbihdr.hdr_name = niftihdr.fname;
  cbihdr.img_name = niftihdr.iname;
  cbihdr.endian = fileEndianness(niftihdr.fname);
  cbihdr.sizeof_hdr = 348;
  cbihdr.dim_info = bitand(uint32(niftihdr.freq_dim),3) + bitand(uint32(niftihdr.phase_dim),3) * 4 + ...
      bitand(uint32(niftihdr.slice_dim),3) * 16;
  cbihdr.dim = double(niftihdr.dim');
  cbihdr.intent_ps = double([niftihdr.intent_p1 niftihdr.intent_p2 niftihdr.intent_p3]');
  cbihdr.intent_code = niftihdr.intent_code;
  cbihdr.datatype = datatypeValue(niftihdr.datatype); 
  cbihdr.bitpix = 8 * niftihdr.nbyper;
  cbihdr.slice_start = niftihdr.slice_start;
  cbihdr.pixdim = double(niftihdr.pixdim');
  cbihdr.vox_offset = single(niftihdr.iname_offset);
  cbihdr.scl_slope = niftihdr.scl_slope;
  cbihdr.scl_inter = niftihdr.scl_inter;
  cbihdr.slice_end = niftihdr.slice_end;
  cbihdr.slice_code = niftihdr.slice_code;
  cbihdr.xyzt_units = bitand(uint32(niftihdr.xyz_units),7) + bitand(uint32(niftihdr.time_units), 56);
  cbihdr.cal_max = niftihdr.cal_max;
  cbihdr.cal_min = niftihdr.cal_min;
  cbihdr.slice_duration = niftihdr.slice_duration;
  cbihdr.toffset = niftihdr.toffset;
  cbihdr.descrip = copyStr(niftihdr.descrip, 80);
  cbihdr.aux_file = copyStr(niftihdr.aux_file, 24);
  cbihdr.qform_code = niftihdr.qform_code;
  cbihdr.sform_code = niftihdr.sform_code;
  cbihdr.quatern_b = niftihdr.quatern_b;
  cbihdr.quatern_c = niftihdr.quatern_c;
  cbihdr.quatern_d = niftihdr.quatern_d;
  cbihdr.qoffset_x = niftihdr.qoffset_x;
  cbihdr.qoffset_y = niftihdr.qoffset_y;
  cbihdr.qoffset_z = niftihdr.qoffset_z;
  cbihdr.srow_x = niftihdr.srow_x;
  cbihdr.srow_y = niftihdr.srow_y;
  cbihdr.srow_z = niftihdr.srow_z;
  cbihdr.intent_name = copyStr(niftihdr.intent_name, 16);
  cbihdr.matlab_datatype = niftihdr.datatype;
  cbihdr.is_analyze = 0;
  switch niftihdr.nifti_type
   case 0
    cbihdr.magic = '    ';
    cbihdr.is_analyze = 1;
   case 1
    cbihdr.magic = 'n+1 ';
    cbihdr.single_file = 1;
   case 2
    cbihdr.magic = 'ni1 ';
   otherwise
    cbihdr.magic = '    ';
  end
  
  cbihdr.qform44 = niftihdr.qto_xyz;
  cbihdr.sform44 = niftihdr.sto_xyz;

return
  
function val = datatypeValue(datatype)

  switch lower(datatype)
   case 'uint8'
    val = int32(2);
   case {'int16', 'short'}
    val = int32(4);
   case {'int32', 'int'}
    val = int32(8);
   case {'single', 'float'}
    val = int32(16);
   case 'complex'
    val = int32(32);
   case 'double'
    val = int32(64);
   case 'RGB'
    val = int32(128);
   case 'int8'
    val = int32(256);
   case {'uint16', 'ushort'}
    val = int32(512);
   case {'uint32', 'uint'}
    val = int32(768);
   case 'int64'
    val = int32(1024);
   case 'uint64'
    val = int32(1280);
   case 'float128'
    val = int32(1536);
   case 'complex128'
    val = int32(1792);
   case 'complex256'
    val = int32(2048);
   otherwise
    error([val,' is not one of valid data types: uint8, uint16, ushort, uint32, uint, uint64, ' ...
           'int8, int16, short, int32, int, int64, single, float, double, float128, RGB, complex, ' ...
           'complex128, complex256']); 
  end

return

function endian = fileEndianness(fname)
  
  % If file does not exist, set to 'native'
    
  if ~exist(fname, 'file')
    endian = 'native';
  else
    % Open as big endian and read first value (header size), if it isn't 348, it must be little
    
    hsize = 0;
    endian = 'b';
    while hsize ~= 348
      fid = fopen(fname, 'r', endian);
      hsize = fread(fid, 1, 'int32');
      fclose(fid);
      if hsize ~= 348
        if endian == 'b'
          endian = 'l';
        else
          error('Illegal file format');
        end
      end
    end
  end
  
return
  
function str = copyStr(src, msiz)
  str = char(zeros(1,msiz));
  slen = min(msiz, length(src));
  if slen > 0
    str(1:slen) = src(1:slen);
  end
  
return
  