function hdr = cbiParseCBIHeader(cbihdr)
  
% Take an mrUtils header (cbihdr) and generate a niftimatlab header (hdr)
  
  hdr = niftifile(cbihdr.hdr_name);
  hdr.freq_dim = bitand(cbihdr.dim_info, 3);
  hdr.phase_dim = bitand(cbihdr.dim_info, 12) / 4;
  hdr.slice_dim = bitand(cbihdr.dim_info, 48) / 16;
  hdr.dim(2:cbihdr.dim(1)) = cbihdr.dim(2:cbihdr.dim(1));
  hdr.pixdim(2:cbihdr.dim(1)) = cbihdr.pixdim(2:cbihdr.dim(1));
  hdr.intent_p1 = cbihdr.intent_ps(1);
  hdr.intent_p2 = cbihdr.intent_ps(2);
  hdr.intent_p3 = cbihdr.intent_ps(3);
  hdr.intent_code = cbihdr.intent_code;
  hdr.slice_start = cbihdr.slice_start;
  hdr.iname_offset = cbihdr.vox_offset;
  hdr.scl_slope = cbihdr.scl_slope;
  hdr.scl_inter = cbihdr.scl_inter;
  hdr.slice_end = cbihdr.slice_end;
  hdr.slice_code = cbihdr.slice_code;
  hdr.xyz_units = bitand(cbihdr.xyzt_units, 7);
  hdr.time_units = bitand(cbihdr.xyzt_units, 56);
  hdr.cal_max = cbihdr.cal_max;
  hdr.cal_min = cbihdr.cal_min;
  hdr.slice_duration = cbihdr.slice_duration;
  hdr.toffset = cbihdr.toffset;
  hdr.descrip = cbihdr.descrip(1:79);
  hdr.aux_file = cbihdr.aux_file(1:23);
  hdr.qform_code = cbihdr.qform_code;
  hdr.sform_code = cbihdr.sform_code;
  hdr.quatern_b = cbihdr.quatern_b;
  hdr.quatern_c = cbihdr.quatern_c;
  hdr.quatern_d = cbihdr.quatern_d;
  hdr.qoffset_x = cbihdr.qoffset_x;
  hdr.qoffset_y = cbihdr.qoffset_y;
  hdr.qoffset_z = cbihdr.qoffset_z;
  hdr.intent_name = cbihdr.intent_name(1:15);
  hdr.datatype = cbihdr.matlab_datatype;
  hdr.qto_xyz = cbihdr.qform44;
  hdr.sto_xyz = cbihdr.sform44;

  return
