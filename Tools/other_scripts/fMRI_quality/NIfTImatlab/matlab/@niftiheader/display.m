function display(a)
% display(a)
%
% Display a niftiheader object
  format compact;
  fprintf('ndim\t\t: %d\n', a.ndim);
  fprintf('nx\t\t: %d\n', a.nx);
  fprintf('ny\t\t: %d\n', a.ny);
  fprintf('nz\t\t: %d\n', a.nz);
  fprintf('nt\t\t: %d\n', a.nt);
  fprintf('nu\t\t: %d\n', a.nu);
  fprintf('nv\t\t: %d\n', a.nv);
  fprintf('nw\t\t: %d\n', a.nw);
  
  fprintf('dim\t\t: [')
  for index = 1:7
    fprintf('%d, ', a.dim(index));
  end
  fprintf('%d]\n', a.dim(8));

  fprintf('nvox\t\t: %d\n', a.nvox);
  fprintf('nbyper\t\t: %d\n', a.nbyper);

  % display data types symbolically
    
  switch a.datatype
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
    
  fprintf('datatype\t: %s\n', val);
  
  fprintf('dx\t\t: %f\n', a.dx);
  fprintf('dy\t\t: %f\n', a.dy);
  fprintf('dz\t\t: %f\n', a.dz);
  fprintf('dt\t\t: %f\n', a.dt);
  fprintf('du\t\t: %f\n', a.du);
  fprintf('dv\t\t: %f\n', a.dv);
  fprintf('dw\t\t: %f\n', a.dw);

  fprintf('pixdim\t\t: [')
  for index = 1:7
    fprintf('%f, ', a.pixdim(index));
  end
  fprintf('%f]\n', a.pixdim(8));

  fprintf('scl_slope\t: %f\n', a.scl_slope);
  fprintf('scl_inter\t: %f\n', a.scl_inter);
  fprintf('cal_min\t\t: %f\n', a.cal_min);
  fprintf('cal_max\t\t: %f\n', a.cal_max);
  fprintf('qform_code\t: %d\n', a.qform_code);
  fprintf('sform_code\t: %d\n', a.sform_code);
  fprintf('freq_dim\t: %d\n', a.freq_dim);
  fprintf('phase_dim\t: %d\n', a.phase_dim);
  fprintf('slice_dim\t: %d\n', a.slice_dim);
  fprintf('slice_code\t: %d\n', a.slice_code);
  fprintf('slice_start\t: %d\n', a.slice_start);
  fprintf('slice_end\t: %d\n', a.slice_end);
  fprintf('slice_duration\t: %f\n', a.slice_duration);
  fprintf('quatern_b\t: %f\n', a.quatern_b);
  fprintf('quatern_c\t: %f\n', a.quatern_c);
  fprintf('quatern_d\t: %f\n', a.quatern_d);
  fprintf('qoffset_x\t: %f\n', a.qoffset_x);
  fprintf('qoffset_y\t: %f\n', a.qoffset_y);
  fprintf('qoffset_z\t: %f\n', a.qoffset_z);
  fprintf('qfac\t\t: %f\n', a.qfac);
  
  fprintf('qto_xyz\t\t: [')
  for indexout = 1:3
    fprintf('(')
    for indexin = 1:3
      fprintf('%f, ', a.qto_xyz(indexout, indexin));
    end
    fprintf('%f)\n\t\t   ', a.qto_xyz(indexout, 4));
  end
  fprintf('(')
  for indexin = 1:3
    fprintf('%f, ', a.qto_xyz(4, indexin));
  end
  fprintf('%f)]\n', a.qto_xyz(4, 4));
  
  fprintf('qto_ijk\t\t: [')
  for indexout = 1:3
    fprintf('(')
    for indexin = 1:3
      fprintf('%f, ', a.qto_ijk(indexout, indexin));
    end
    fprintf('%f)\n\t\t   ', a.qto_ijk(indexout, 4));
  end
  fprintf('(')
  for indexin = 1:3
    fprintf('%f, ', a.qto_ijk(4, indexin));
  end
  fprintf('%f)]\n', a.qto_ijk(4, 4));
  
  fprintf('sto_xyz\t\t: [')
  for indexout = 1:3
    fprintf('(')
    for indexin = 1:3
      fprintf('%f, ', a.sto_xyz(indexout, indexin));
    end
    fprintf('%f)\n\t\t   ', a.sto_xyz(indexout, 4));
  end
  fprintf('(')
  for indexin = 1:3
    fprintf('%f, ', a.sto_xyz(4, indexin));
  end
  fprintf('%f)]\n', a.sto_xyz(4, 4));
  
  fprintf('sto_ijk\t\t: [')
  for indexout = 1:3
    fprintf('(')
    for indexin = 1:3
      fprintf('%f, ', a.sto_ijk(indexout, indexin));
    end
    fprintf('%f)\n\t\t   ', a.sto_ijk(indexout, 4));
  end
  fprintf('(')
  for indexin = 1:3
    fprintf('%f, ', a.sto_ijk(4, indexin));
  end
  fprintf('%f)]\n', a.sto_ijk(4, 4));
  
  fprintf('toffset\t\t: %f\n', a.toffset);
  fprintf('xyz_units\t: %d\n', a.xyz_units);
  fprintf('time_units\t: %d\n', a.time_units);
  fprintf('nifti_type\t: ');
  
  switch a.nifti_type
   case 0
    fprintf('ANALYZE 7.5\n');
   case 1
    fprintf('NIfTI-1 Single file\n');
   case 2
    fprintf('NIfTI-1 Dual File\n');
   case 3
    fprintf('NIfTI-ASCII\n');
  end
  
  fprintf('intent_code\t: %d\n', a.intent_code);
  fprintf('intent_p1\t: %f\n', a.intent_p1);
  fprintf('intent_p2\t: %f\n', a.intent_p2);
  fprintf('intent_p3\t: %f\n', a.intent_p3);
  fprintf('intent_name\t: %s\n', a.intent_name);
  fprintf('descrip\t\t: %s\n', a.descrip);
  fprintf('aux_file\t: %s\n', a.aux_file);
  fprintf('fname\t\t: %s\n', a.fname);
  fprintf('iname\t\t: %s\n', a.iname);
  fprintf('iname_offset\t: %d\n', a.iname_offset);
  fprintf('swapsize\t: %d\n', a.swapsize);
  fprintf('byteorder\t: %d\n', a.byteorder);
  fprintf('num_ext\t\t: %d\n', a.num_ext);
  for extension = 1:a.num_ext
    fprintf('Extension %d: size %d, code %d\n', extension, ...
            a.ext_list(extension).esize, a.ext_list(extension).ecode);
  end
  

