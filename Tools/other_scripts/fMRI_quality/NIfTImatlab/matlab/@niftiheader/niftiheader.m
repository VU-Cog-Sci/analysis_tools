function [nhdr] = niftiheader(name)
% [nhdr] = niftiheader(name)
%
% niftiheader Constructor
%
% Create a niftiheader object
%
%   nhdr = nifitiheader()	Create default niftiheader object
%   
%   nhdr = niftiheader(NAME)	Create default niftiheader object and set file to NAME
%				If extension is present, nifti type will be inferred with
%				precendence given to NIfTI dual file over ANALYZE 7.5.
%				If no extension given, type will have to be set with 'type'
%				assignment. 
%				
% Return: nifti header object
%
% Methods: 

  % Basic error checking
    
  if nargin > 1
    error('Easy cowboy, I cannot take more than 1 argument !');
  end
  
  if nargin == 1
    if ~ischar(name)
      error('NAME must be a string');
    end
  end
  
  if nargout > 1
    error('What am i supposed to do with all those output arguments ?');
  end

  % Initialize with blank values
    
  nhdr = Defaults;
  
  nhdr = class(nhdr, 'niftiheader');

  % Store path and name
    
  if nargin == 1
    nhdr = set(nhdr, 'name', name);
  end
  
function nhdr = Defaults()
  
  nhdr.ndim = int32(1);
  nhdr.nx = int32(1);
  nhdr.ny = int32(1);
  nhdr.nz = int32(1);
  nhdr.nt = int32(1);
  nhdr.nu = int32(1);
  nhdr.nv = int32(1);
  nhdr.nw = int32(1);
  nhdr.dim = [int32(1) int32(1) int32(1) int32(1) int32(1) int32(1) int32(1) int32(1)];
  nhdr.nvox = int32(1);
  nhdr.nbyper = int32(4);
  nhdr.datatype = int32(8);
  nhdr.dx = single(1);
  nhdr.dy = single(1);
  nhdr.dz = single(1);
  nhdr.dt = single(1);
  nhdr.du = single(1);
  nhdr.dv = single(1);
  nhdr.dw = single(1);
  nhdr.pixdim = [single(1) single(1) single(1) single(1) single(1) single(1) single(1) single(1)];
  nhdr.scl_slope = single(0);
  nhdr.scl_inter = single(0);
  nhdr.cal_min = single(0);
  nhdr.cal_max = single(0);
  nhdr.qform_code = int32(0);
  nhdr.sform_code = int32(0);
  nhdr.freq_dim = int32(0);
  nhdr.phase_dim = int32(0);
  nhdr.slice_dim = int32(0);
  nhdr.slice_code = int32(0);
  nhdr.slice_start = int32(0);
  nhdr.slice_end = int32(0);
  nhdr.slice_duration = single(0);
  nhdr.quatern_b = single(0);
  nhdr.quatern_c = single(0);
  nhdr.quatern_d = single(0);
  nhdr.qoffset_x = single(0);
  nhdr.qoffset_y = single(0);
  nhdr.qoffset_z = single(0);
  nhdr.qfac = single(0);
  nhdr.qto_xyz = zeros(4);
  nhdr.qto_ijk = zeros(4);
  nhdr.sto_xyz = zeros(4);
  nhdr.sto_ijk = zeros(4);
  nhdr.toffset = single(0);
  nhdr.xyz_units = int32(0);
  nhdr.time_units = int32(0);
  nhdr.nifti_type = int32(2);
  nhdr.intent_code = int32(0);
  nhdr.intent_p1 = single(0);
  nhdr.intent_p2 = single(0);
  nhdr.intent_p3 = single(0);
  nhdr.intent_name = '';
  nhdr.descrip = '';
  nhdr.aux_file = '';
  nhdr.fname = '';
  nhdr.iname = '';
  nhdr.iname_offset = int32(0);
  nhdr.swapsize = int32(0);
  nhdr.byteorder = int32(0);
  nhdr.num_ext = int32(0);
  nhdr.ext_list(1).esize = int32(0);
  nhdr.ext_list(1).ecode = int32(0);
  nhdr.ext_list(1).edata = ' ';
  
    
