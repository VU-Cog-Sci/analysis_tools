function nhdr = set(nhdr,varargin)
% nhdr = set(nhdr,propname,propval,...)
%
% Set niftiheader properties and return the updated object

  if ~isa(nhdr,'niftiheader'),
   % Call built-in SET. 
   builtin('set', nhdr, varargin{:});
   return
  end
  
  if nargin == 1
    if nargout == 0
      DisplayOptions;
    end
  else
    propertyArgIn = varargin;
    while length(propertyArgIn) > 0
      prop = propertyArgIn{1};
      if length(prop) == 0
        return
      end
      if length(propertyArgIn) == 1
        switch char(prop)
         case 'datatype'
          datatype(nhdr);
         case 'compressed'
          compressed;
         case {'type', 'nifti_type'}
          type;
         case {'ndim', 'nx', 'ny', 'nz', 'nt', 'nu', 'nv', 'nw', 'nvox', 'nbyper', ...
               'qform_code', 'sform', 'freq_dim', 'phase_dim', 'slice_dim', 'slice_code', 'slice_start', ...
               'slice_end', 'xyz_units', 'time_units', 'intent', 'iname_offset', 'num_ext'}
          fprintf(1, '  %s : int32\n', char(prop));
         case 'dim'
          fprintf(1, '  dim : 8 x int32\n');
         case {'dx', 'dy', 'dz', 'dt', 'du', 'dv', 'dw', 'scl_inter', 'scl_slope', 'cal_max', 'cal_min', ...
               'slice_duration', 'quatern_b', 'quatern_c', 'quatern_d', 'qoffset_x', 'qoffset_y', 'qoffset_z', ...
               'qfac', 'toffset', 'intent_code', 'intent_name', 'intent_p1', 'intent_p2', 'intent_p3', 'pixdim'}
          fprintf(1, '  %s : single\n', char(prop));
         case {'intent', 'descrip', 'aux_file', 'fname', 'iname'}
          fprintf(1, '  %s : string\n', char(prop));
         case {'qto_xyz', 'qto_ijk', 'sto_xyz', 'sto_ijk'}
          fprintf(1, '  %s : 4x4 double\n', char(prop));
         otherwise
          DisplayOptions;
        end
        return
      else
        val = propertyArgIn{2};
        propertyArgIn = propertyArgIn(3:end);
        switch prop
         case {'ndim', 'nvox'}
          error('Value of %s will be set automatically when changing nx, ny, nz, nt, nu, nv, nw', ...
                prop);
         case 'dim'
          index = val{1};
          val = propertyArgIn{1}{1};
          propertyArgIn = propertyArgIn(2:end);
          while length(index) > 0
            switch index(1)
             case 1
              error('Value of dim(1) is set automatically when setting the other entries.');
             case 2
              nhdr = set(nhdr, 'nx', val(1));
             case 3
              nhdr = set(nhdr, 'ny', val(1));
             case 4
              nhdr = set(nhdr, 'nz', val(1));
             case 5
              nhdr = set(nhdr, 'nt', val(1));
             case 6
              nhdr = set(nhdr, 'nu', val(1));
             case 7
              nhdr = set(nhdr, 'nv', val(1));
             case 8
              nhdr = set(nhdr, 'nw', val(1));
             otherwise
              error('Invalid index %d', index(1));
            end
            index = index(2:end);
            val = val(2:end);
          end
         case 'nx'
          nhdr.nx = int32(val);
          nhdr = setDims(nhdr, 2, val);
         case 'ny'
          nhdr.ny = int32(val);
          nhdr = setDims(nhdr, 3, val);
         case 'nz'
          nhdr.nz = int32(val);
          nhdr = setDims(nhdr, 4, val);
         case 'nt'
          nhdr.nt = int32(val);
          nhdr = setDims(nhdr, 5, val);
         case 'nu'
          nhdr.nu = int32(val);
          nhdr = setDims(nhdr, 6, val);
         case 'nv'
          nhdr.nv = int32(val);
          nhdr = setDims(nhdr, 7, val);
         case 'nw'
          nhdr.nw = int32(val);
          nhdr = setDims(nhdr, 8, val);
         case 'datatype'
          nhdr = datatype(nhdr, val);
         case 'dx'
          nhdr.dx = single(val);
          nhdr.pixdim(2) = single(val);
         case 'dy'
          nhdr.dy = single(val);
          nhdr.pixdim(3) = single(val);
         case 'dz'
          nhdr.dz = single(val);
          nhdr.pixdim(4) = single(val);
         case 'dt'
          nhdr.dt = single(val);
          nhdr.pixdim(5) = single(val);
         case 'du'
          nhdr.du = single(val);
          nhdr.pixdim(6) = single(val);
         case 'dv'
          nhdr.dv = single(val);
          nhdr.pixdim(7) = single(val);
         case 'dw'
          nhdr.dw = single(val);
          nhdr.pixdim(8) = single(val);
         case 'pixdim'
          index = val{1};
          val = propertyArgIn{1}{1};
          propertyArgIn = propertyArgIn(2:end);
          while length(index) > 0
            switch index(1)
             case 2
              nhdr = set(nhdr, 'dx', val(1));
             case 3
              nhdr = set(nhdr, 'dy', val(1));
             case 4
              nhdr = set(nhdr, 'dz', val(1));
             case 5
              nhdr = set(nhdr, 'dt', val(1));
             case 6
              nhdr = set(nhdr, 'du', val(1));
             case 7
              nhdr = set(nhdr, 'dv', val(1));
             case 8
              nhdr = set(nhdr, 'dw', val(1));
             otherwise
              error('Invalid index %d', index(1));
            end
            index = index(2:end);
            val = val(2:end);
          end
         case 'scl_slope'
          nhdr.scl_slope = single(val);
         case 'scl_inter'
          nhdr.scl_inter = single(val);
         case 'cal_min'
          nhdr.cal_min = single(val);
         case 'cal_max'
          nhdr.cal_max = single(val);
         case 'qform_code'
          nhdr.qform_code = int32(val);
          if val == 0
            nhdr.qto_xyz = zeros(4);
            nhdr.qto_ijk = zeros(4);
          end
         case 'sform_code'
          nhdr.sform_code = int32(val);
          if val == 0
            nhdr.sto_xyz = zeros(4);
            nhdr.sto_ijk = zeros(4);
          end
         case 'freq_dim'
          nhdr.freq_dim = int32(val);
         case 'phase_dim'
          nhdr.phase_dim = int32(val);
         case 'slice_dim'
          nhdr.slice_dim = int32(val);
         case 'slice_code'
          nhdr.slice_code = int32(val);
         case 'slice_start'
          nhdr.slice_start = int32(val);
         case 'slice_end'
          nhdr.slice_end = int32(val);
         case 'slice_duration'
          nhdr.slice_duration = single(val);
         case 'quatern_b'
          nhdr.quatern_b = single(val);
         case 'quatern_c'
          nhdr.quatern_c = single(val);
         case 'quatern_d'
          nhdr.quatern_d = single(val);
         case 'quatern'
          nhdr.quatern_b = single(val(1));
          nhdr.quatern_c = single(val(2));
          nhdr.quatern_d = single(val(3));
         case 'qoffset_x'
          nhdr.qoffset_x = single(val);
         case 'qoffset_y'
          nhdr.qoffset_y = single(val);
         case 'qoffset_z'
          nhdr.qoffset_z = single(val);
         case 'qoffset'
          nhdr.qoffset_x = single(val(1));
          nhdr.qoffset_y = single(val(2));
          nhdr.qoffset_z = single(val(3));
         case 'qfac'
          nhdr.qfac = single(val);
          if (abs(nhdr.qfac)<1e-10)
            disp(['Invalid qfac (' num2str(nhdr.qfac) '). Assuming qfac = 1']);
            nhdr.qfac = single(1);
          end
          if nhdr.qfac >= 0
            nhdr.pixdim(1) = single(1);
          else
            nhdr.pixdim(1) = single(-1);
          end
         case 'qto_xyz'
          nhdr.qto_xyz = val;
          if det(val) ~= 0
            nhdr.qto_ijk = inv(val);
          else
            nhdr.qto_ijk = zeros(4);
          end
         case 'qto_ijk'
          nhdr.qto_ijk = val;
          if det(val) ~= 0
            nhdr.qto_xyz = inv(val);
          else
            nhdr.qto_xyz = zeros(4);
          end
         case 'sto_xyz'
          nhdr.sto_xyz = val;
          if det(val) ~= 0
            nhdr.sto_ijk = inv(val);
          else
            nhdr.sto_ijk = zeros(4);
          end
         case 'sto_ijk'
          nhdr.sto_ijk = val;
          if det(val) ~= 0
            nhdr.sto_xyz = inv(val);
          else
            nhdr.sto_ijk = zeros(4);
          end
         case 'toffset'
          nhdr.toffset = single(val);
         case 'xyz_units'
          nhdr.xyz_units = int32(val);
         case 'time_units'
          nhdr.time_units = int32(val);
         case 'intent_p1'
          nhdr.intent_p1 = single(val);
         case 'intent_p2'
          nhdr.intent_p2 = single(val);
         case 'intent_p3'
          nhdr.intent_p3 = single(val);
         case 'intent_name'
          nhdr.intent_name = val;
         case 'intent_code'
          nhdr.intent_code = int32(val);
         case 'descrip'
          nhdr.descrip = val;
         case 'aux_file'
          nhdr.aux_file = val;
         case {'fname', 'iname', 'name'}
          [nhdr.fname, nhdr.iname, nhdr.nifti_type] = name(val, nhdr.nifti_type);
         case {'type', 'nifti_type'}
          [nhdr.fname, nhdr.iname, nhdr.nifti_type] = type(nhdr.fname, val);
         case 'compressed'
          if val
            [nhdr.fname, nhdr.iname] = compressed(nhdr.fname, nhdr.iname);
          else
            [nhdr.fname, nhdr.iname] = type(nhdr.fname, nhdr.nifti_type);
          end
         case 'iname_offset'
          nhdr.iname_offset = int32(val);
         case 'byteorder'
          nhdr.byteorder = int32(val);
         case 'num_ext'
          nhdr.num_ext = int32(val);
         case 'ext_list.esize'
          nhdr.ext_list.esize = int32(val);
         case 'ext_list.ecode'
          nhdr.ext_list.ecode = int32(val);
         otherwise
          error([prop,' Is not a settable niftiheader field'])
        end
        
        % If any of the parameters involved in the transformation matrix have changed, 
        % recompute it

        switch prop
         case {'quatern', 'quatern_b', 'quatern_c', 'quatern_d', ...
               'qoffset', 'qoffset_x', 'qoffset_y', 'qoffset_z', ...
               'qfac', 'dx', 'dy', 'dz', 'dt', 'du', 'dv', 'dw'}
          nhdr.qto_xyz = quatToMat44(nhdr);
          nhdr.qto_ijk = inv(nhdr.qto_xyz);
         case {'qto_xyz', 'qto_ijk'}
          [quatern, nhdr.qfac, pixdim] = mat44ToQuat(nhdr.qto_xyz);
          nhdr.quatern_b = single(quatern(1));
          nhdr.quatern_c = single(quatern(2));
          nhdr.quatern_d = single(quatern(3));
          nhdr.qoffset_x = single(nhdr.qto_xyz(1,4));
          nhdr.qoffset_y = single(nhdr.qto_xyz(2,4));
          nhdr.qoffset_z = single(nhdr.qto_xyz(3,4));
          if nhdr.qfac >= 0
            nhdr.pixdim(1) = single(1);
          else
            nhdr.pixdim(1) = single(-1);
          end
          nhdr.pixdim(2) = single(pixdim(1));
          nhdr.pixdim(3) = single(pixdim(2));
          nhdr.pixdim(4) = single(pixdim(3));
          nhdr.dx = single(pixdim(1));
          nhdr.dy = single(pixdim(2));
          nhdr.dz = single(pixdim(3));
          nhdr.qform_code = int32(1);
        end
      end
    end
  end
  
function DisplayOptions
  fprintf(1, '  ndim            : int32\n');
  fprintf(1, '  nx              : int32\n');
  fprintf(1, '  ny              : int32\n');
  fprintf(1, '  nz              : int32\n');
  fprintf(1, '  nt              : int32\n');
  fprintf(1, '  nu              : int32\n');
  fprintf(1, '  nv              : int32\n');
  fprintf(1, '  nw              : int32\n');
  fprintf(1, '  dim             : 8 x  int32\n');
  fprintf(1, '  nvox            : int32\n');
  fprintf(1, '  nbyper          : int32\n');
  fprintf(1, '  datatype        : [ uint8 | uint16 | uint32 | uint | uint64 | int8');
  fprintf(1, ' | int16 | short | int32 | int | int64 | single | float | double | float128');
  fprintf(1, ' | RGB | complex | complex128 | complex256]\n');
  fprintf(1, '  dx              : single\n');
  fprintf(1, '  dy              : single\n');
  fprintf(1, '  dz              : single\n');
  fprintf(1, '  dt              : single\n');
  fprintf(1, '  du              : single\n');
  fprintf(1, '  dv              : single\n');
  fprintf(1, '  dw              : single\n');
  fprintf(1, '  pixdim          : 8 x single\n');
  fprintf(1, '  scl_slope       : single\n');
  fprintf(1, '  scl_inter       : single\n');
  fprintf(1, '  cal_min         : single\n');
  fprintf(1, '  cal_max         : single\n');
  fprintf(1, '  qform_code      : int32\n');
  fprintf(1, '  sform_code      : int32\n');
  fprintf(1, '  freq_dim        : int32\n');
  fprintf(1, '  phase_dim       : int32\n');
  fprintf(1, '  slice_dim       : int32\n');
  fprintf(1, '  slice_code      : int32\n');
  fprintf(1, '  slice_start     : int32\n');
  fprintf(1, '  slice_end       : int32\n');
  fprintf(1, '  slice_duration  : single\n');
  fprintf(1, '  quatern_b       : single\n');
  fprintf(1, '  quatern_c       : single\n');
  fprintf(1, '  quatern_d       : single\n');
  fprintf(1, '  qoffset_x       : single\n');
  fprintf(1, '  qoffset_y       : single\n');
  fprintf(1, '  qoffset_z       : single\n');
  fprintf(1, '  qfac            : single\n');
  fprintf(1, '  qto_xyz         : 4x4 double \n');
  fprintf(1, '  qto_ijk         : 4x4 double \n');
  fprintf(1, '  sto_xyz         : 4x4 double \n');
  fprintf(1, '  sto_ijk         : 4x4 double \n');
  fprintf(1, '  toffset         : single\n');
  fprintf(1, '  xyz_units       : int32\n');
  fprintf(1, '  time_units      : int32\n');
  fprintf(1, '  nifti_type      : [ single | dual | analyze | ascii ]\n');
  fprintf(1, '  intent_code     : int32\n');
  fprintf(1, '  intent_p1       : single\n');
  fprintf(1, '  intent_p2       : single\n');
  fprintf(1, '  intent_p3       : single\n');
  fprintf(1, '  intent_name     : string\n');
  fprintf(1, '  descrip         : string\n');
  fprintf(1, '  aux_file        : string\n');
  fprintf(1, '  fname           : string\n');
  fprintf(1, '  iname           : string\n');
  fprintf(1, '  iname_offset    : int32\n');
  fprintf(1, '  num_ext         : int32\n');
  fprintf(1, '  compressed      : [ true | false ]\n');

function nhdr = setDims(nhdr, index, val)
  
  nhdr.dim(index) = int32(val);
  for dim = 8:-1:2, if nhdr.dim(dim) > 1, break, end, end
  nhdr.dim(1) = int32(dim-1);
  nhdr.ndim = int32(dim-1);
  nhdr.nvox = int32(prod(double(nhdr.dim(2:dim))));
  