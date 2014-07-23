function val = get(nhdr, propName)
% val = get(nhdr, propName)
%
% Get niftiheader property
  switch propName
   case 'ndim'
    val = nhdr.ndim;
   case 'nx'
    val = nhdr.nx;
   case 'ny'
    val = nhdr.ny;
   case 'nz'
    val = nhdr.nz;
   case 'nt'
    val = nhdr.nt;
   case 'nu'
    val = nhdr.nu;
   case 'nv'
    val = nhdr.nv;
   case 'nw'
    val = nhdr.nw;
   case 'dim'
    val = nhdr.dim;
   case 'nvox'
    val = nhdr.nvox;
   case 'nbyper'
    val = nhdr.nbyper;
   
    % display data types symbolically
    
   case 'datatype'
    switch nhdr.datatype
     case 1
      val = 'binary';
     case 2
      val = 'uint8';
     case 4
      val = 'int16';
     case 8
      val = 'int32';
     case 16
      val = 'single';
     case 32
      val = 'complex';
     case 64
      val = 'double';
     case 128
      val = 'RGB';
     case 256
      val = 'int8';
     case 512
      val = 'uint16';
     case 768
      val = 'uint32';
     case 1024
      val = 'int64';
     case 1280
      val = 'uint64';
     case 1536
      val = 'float128';  
     case 1792
      val = 'complex128';  
     case 2048
      val = 'complex256';  
     otherwise
      val = 'UNDEFINED';
    end
    
   case 'dx'
    val = nhdr.dx;
   case 'dy'
    val = nhdr.dy;
   case 'dz'
    val = nhdr.dz;
   case 'dt'
    val = nhdr.dt;
   case 'du'
    val = nhdr.du;
   case 'dv'
    val = nhdr.dv;
   case 'dw'
    val = nhdr.dw;
   case 'pixdim'
    val = nhdr.pixdim;
   case 'scl_slope'
    val = nhdr.scl_slope;
   case 'scl_inter'
    val = nhdr.scl_inter;
   case 'cal_min'
    val = nhdr.cal_min;
   case 'cal_max'
    val = nhdr.cal_max;
   case 'qform_code'
    val = nhdr.qform_code;
   case 'sform_code'
    val = nhdr.sform_code;
   case 'freq_dim'
    val = nhdr.freq_dim;
   case 'phase_dim'
    val = nhdr.phase_dim;
   case 'slice_dim'
    val = nhdr.slice_dim;
   case 'slice_code'
    val = nhdr.slice_code;
   case 'slice_start'
    val = nhdr.slice_start;
   case 'slice_end'
    val = nhdr.slice_end;
   case 'slice_duration'
    val = nhdr.slice_duration;
   case 'quatern_b'
    val = nhdr.quatern_b;
   case 'quatern_c'
    val = nhdr.quatern_c;
   case 'quatern_d'
    val = nhdr.quatern_d;
   case 'qoffset_x'
    val = nhdr.qoffset_x;
   case 'qoffset_y'
    val = nhdr.qoffset_y;
   case 'qoffset_z'
    val = nhdr.qoffset_z;
   case 'qfac'
    val = nhdr.qfac;
   case 'qto_xyz'
    val = nhdr.qto_xyz;
   case 'qto_ijk'
    val = nhdr.qto_ijk;
   case 'sto_xyz'
    val = nhdr.sto_xyz;
   case 'sto_ijk'
    val = nhdr.sto_ijk;
   case 'toffset'
    val = nhdr.toffset;
   case 'xyz_units'
    val = nhdr.xyz_units;
   case 'time_units'
    val = nhdr.time_units;
   case 'nifti_type'
    val = nhdr.nifti_type;
   case 'intent_p1'
    val = nhdr.intent_p1;
   case 'intent_p2'
    val = nhdr.intent_p2;
   case 'intent_p3'
    val = nhdr.intent_p3;
   case 'intent_name'
    val = nhdr.intent_name;
   case 'intent_code'
    val = nhdr.intent_code;
   case 'descrip'
    val = nhdr.descrip;
   case 'aux_file'
    val = nhdr.aux_file;
   case 'fname'
    val = nhdr.fname;
   case 'iname'
    val = nhdr.iname;
   case 'iname_offset'
    val = nhdr.iname_offset;
   case 'swapsize'
    val = nhdr.swapsize;
   case 'byteorder'
    val = nhdr.byteorder;
   case 'num_ext'
    val = nhdr.num_ext;
   case 'ext_list'
    val = nhdr.ext_list;
   case 'ext_list.esize'
    val = nhdr.ext_list.esize;
   case 'ext_list.ecode'
    val = nhdr.ext_list.ecode;
   case 'srow_x'
    val = nhdr.sto_xyz(1,:);
   case 'srow_y'
    val = nhdr.sto_xyz(2,:);
   case 'srow_z'
    val = nhdr.sto_xyz(3,:);    
   otherwise
    error([propName,' Is not a valid niftiheader field'])
  end
