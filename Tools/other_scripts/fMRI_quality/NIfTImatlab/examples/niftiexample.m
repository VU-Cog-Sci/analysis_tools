fname = 'example.nii';

% Create the nifti file object
nfdin = niftifile(fname);

% Open the nifti file for reading
nfdin = fopen(nfdin,'read');
% nfdin now contains the header information

% initialize the data array
% Matlab is column major so order the data y,x
data = zeros(nfdin.ny, nfdin.nx, nfdin.nz, nfdin.nt);
 
% Read the data one volume at a time
% This could be done in lots of ways 
for i = 1:nfdin.nt
  [nfdin, databuff] = fread(nfdin, nfdin.nx*nfdin.ny*nfdin.nz);
  % Note that matlab assumes column major format
  % NIFTI is stored so x increases fastest. We need permute.
  data(:,:,:,i) = permute(reshape(databuff, [nfdin.nx nfdin.ny nfdin.nz]), [2 1 3]);
end

% This closes the file, but lets us keep using the header
nfdin = fclose(nfdin);

% Draw a slice
imagesc(data(:,:,16,1))
axis image
axis ij
% The upper left hand corner is pixel (1,1)
% y goes top to bottom
% x goes left to right

% Do some math.
M = 4.0*mean(data,4);

% Create a new file handle with a copy of the other one's header
nfdout = niftifile('outtest',nfdin);
nfdout.descrip = 'modified';

% Make sure we have the correct data type stored
nfdout.datatype = class(M);

% The input file was 4D, we took the mean across time. Change the number
% of dimensions and total number of points accordingly.
nfdout.nvox = nfdout.nx*nfdout.ny*nfdout.nz; 
nfdout.nt = 1;

% Open for writing
nfdout = fopen(nfdout,'write'); % at this point the header is written

% Write out the data
% Remember to put it so that x increases fastest.
wbuff =  reshape( ipermute(M, [2 1 3]), 1, nfdout.nvox);
[nfdout] = fwrite(nfdout, wbuff, nfdout.nvox);

% Close the handle
nfdout = fclose(nfdout);
