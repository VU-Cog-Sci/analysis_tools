function val = get(nfd, varargin)
% val = get(nfd, prop)
%
% Get value of niftifile property
% 
% Parameters: property - name of property
%
% Return: value of property
%
 
  switch varargin{1}
   case 'ndim'
    val = nfd.niftiheader.ndim;
   case 'nx'
    val = nfd.niftiheader.nx;
   case 'ny'
    val = nfd.niftiheader.ny;
   case 'nz'
    val = nfd.niftiheader.nz;
   case 'nt'
    val = nfd.niftiheader.nt;
   case 'nu'
    val = nfd.niftiheader.nu;
   case 'nv'
    val = nfd.niftiheader.nv;
   case 'nw'
    val = nfd.niftiheader.nw;
   case 'dim'
    val = nfd.niftiheader.dim;
   case 'nvox'
    val = nfd.niftiheader.nvox;
   case 'nbyper'
    val = nfd.niftiheader.nbyper;
   case 'datatype'
    val = nfd.niftiheader.datatype;
   case 'dx'
    val = nfd.niftiheader.dx;
   case 'dy'
    val = nfd.niftiheader.dy;
   case 'dz'
    val = nfd.niftiheader.dz;
   case 'dt'
    val = nfd.niftiheader.dt;
   case 'du'
    val = nfd.niftiheader.du;
   case 'dv'
    val = nfd.niftiheader.dv;
   case 'dw'
    val = nfd.niftiheader.dw;
   case 'pixdim'
    val = nfd.niftiheader.pixdim;
   case 'scl_slope'
    val = nfd.niftiheader.scl_slope;
   case 'scl_inter'
    val = nfd.niftiheader.scl_inter;
   case 'cal_min'
    val = nfd.niftiheader.cal_min;
   case 'cal_max'
    val = nfd.niftiheader.cal_max;
   case 'qform_code'
    val = nfd.niftiheader.qform_code;
   case 'sform_code'
    val = nfd.niftiheader.sform_code;
   case 'freq_dim'
    val = nfd.niftiheader.freq_dim;
   case 'phase_dim'
    val = nfd.niftiheader.phase_dim;
   case 'slice_dim'
    val = nfd.niftiheader.slice_dim;
   case 'slice_code'
    val = nfd.niftiheader.slice_code;
   case 'slice_start'
    val = nfd.niftiheader.slice_start;
   case 'slice_end'
    val = nfd.niftiheader.slice_end;
   case 'slice_duration'
    val = nfd.niftiheader.slice_duration;
   case 'quatern_b'
    val = nfd.niftiheader.quatern_b;
   case 'quatern_c'
    val = nfd.niftiheader.quatern_c;
   case 'quatern_d'
    val = nfd.niftiheader.quatern_d;
   case 'qoffset_x'
    val = nfd.niftiheader.qoffset_x;
   case 'qoffset_y'
    val = nfd.niftiheader.qoffset_y;
   case 'qoffset_z'
    val = nfd.niftiheader.qoffset_z;
   case 'qfac'
    val = nfd.niftiheader.qfac;
   case 'qto_xyz'
    val = nfd.niftiheader.qto_xyz;
   case 'qto_ijk'
    val = nfd.niftiheader.qto_ijk;
   case 'sto_xyz'
    val = nfd.niftiheader.sto_xyz;
   case 'sto_ijk'
    val = nfd.niftiheader.sto_ijk;
   case 'toffset'
    val = nfd.niftiheader.toffset;
   case 'xyz_units'
    val = nfd.niftiheader.xyz_units;
   case 'time_units'
    val = nfd.niftiheader.time_units;
   case 'nifti_type'
    val = nfd.niftiheader.nifti_type;
   case 'intent_p1'
    val = nfd.niftiheader.intent_p1;
   case 'intent_p2'
    val = nfd.niftiheader.intent_p2;
   case 'intent_p3'
    val = nfd.niftiheader.intent_p3;
   case 'intent_name'
    val = nfd.niftiheader.intent_name;
   case 'intent_code'
    val = nfd.niftiheader.intent_code;
   case 'descrip'
    val = nfd.niftiheader.descrip;
   case 'aux_file'
    val = nfd.niftiheader.aux_file;
   case 'fname'
    val = nfd.niftiheader.fname;
   case 'iname'
    val = nfd.niftiheader.iname;
   case 'iname_offset'
    val = nfd.niftiheader.iname_offset;
   case 'swapsize'
    val = nfd.niftiheader.swapsize;
   case 'byteorder'
    val = nfd.niftiheader.byteorder;
   case 'num_ext'
    val = nfd.niftiheader.num_ext;
   case 'ext_list'
    val = nfd.niftiheader.ext_list;
   case 'name'
    val = nfd.niftiheader.fname;
   case 'type'
    val = nfd.niftiheader.nifti_type;
   case 'compressed'
    val = nfd.compressed;
   case 'mode'
    val = nfd.mode;
   case 'srow_x'
    val = nfd.niftiheader.srow_x;
   case 'srow_y'
    val = nfd.niftiheader.srow_y;
   case 'srow_z'
    val = nfd.niftiheader.srow_z;
   case 'niftiheader'
    if length(varargin) > 1
      val = get(nfd.niftiheader, varargin{2});
    else
      val = nfd.niftiheader;
    end
   case 'filePos'
    error([varargin{1}, ' is not a public niftifile property']);
   case 'fileId'
    error([varargin{1}, ' is not a public niftifile property']);
   otherwise
    error([varargin{1}, ' is not a valid niftifile property']);
  end

  % If we are asked for an element of an array, try to get it

  if ~strcmp('niftiheader', varargin{1})
    if length(varargin) > 1
      index = varargin{2}{1};
      arglen = length(index);
      if arglen == 0
        error('invalid dimensions');
      elseif arglen == 1
        if index < 1
          error(['invalid value ', num2str(index)]);
        elseif index > length(val)
          error(['"',varargin{1}, '" only has ', num2str(length(val)), ' elements']);
        end
      else
        if index(1) < 1
          error(['invalid lower bound ', num2str(index(1))]);
        elseif index(arglen) > length(val)
          error(['"', varargin{1}, '" only has ', num2str(length(val)), ' elements']);
        end
      end
      
      val = val(index);
    end
  end
  
  
