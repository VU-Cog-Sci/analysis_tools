function cbihdr = cbiReadNiftiHeader(fname)
% cbihdr = cbiReadNiftiHeader(fname)
% 
% Loads the header from a NIFTI-1 file.  

  hdr = niftifile(fname);
  hdr = fopen(hdr, 'read');
  hdr = fclose(hdr);
  cbihdr = cbiCreateNiftiHeader(hdr);
  