function Nz = getNSlices(DICOMheaderInfo)

% getNSlices
%
%       Nz = getNSlices(DICOMheaderInfo)
%
% 'getNSlices' reads the number of slices from the shadow set of a DICOM
%   header.  It has been tested only on Siemens images.
%
% PJV: v.1.1: February 2009

% PJV: v.1.1: February 2009:
%     Correct number of slices for 3D images
% PJV: v.1.0: January 2009


myLongString = char(DICOMheaderInfo.Private_0029_1020');

%% get list of parameters:
start_pos = strfind(myLongString, 'ASCCONV BEGIN');  % points to position of ...
    % ... the 1st letter of "ASCCONV..." in myLongString.
myLongString = myLongString(start_pos+17 : end);  % start of parameters

%% get the size of the slice array
%   (i.e.: no. slices for 2D, no. of slabs for 3D)
start_pos = strfind(myLongString,'sSliceArray.lSize');
        % FYI: 'sSliceArray.lSize' stands for ...
        % ... "structure 'SliceArray', field long 'Size'
% we store in myShortString 51 characters from the 'start_pos',...
% ... to make sure we include the number (there are some spaces, etc):
myShortString = myLongString(start_pos:start_pos+50);
start_pos = strfind(myShortString, '=');  % Search for a "=" in myShortString.
sliceArraySize = sscanf(myShortString(start_pos+2:end), '%i');    % We take ...
                % ... the number after the equal plus spaces...

%% get the dimensionality of the images:
dimensionality = sscanf( DICOMheaderInfo.MRAcquisitionType, '%i' );

%% get the number of images per slab:

if dimensionality==2      % 2D sequence:
  slicesPerSlab = 1;        % 1 slice per "slab"
elseif dimensionality==3  % 3D sequence:
  start_pos = strfind(myLongString,'sKSpace.lImagesPerSlab');
  % we store in myShortString 51 characters from the 'start_pos',...
  % ... to make sure we include the number (there are some spaces, etc):
  myShortString = myLongString(start_pos:start_pos+50);
  start_pos = strfind(myShortString, '=');  % Search for a "=" in myShortString.
  slicesPerSlab = sscanf(myShortString(start_pos+2:end), '%i');    % We take ...
                % ... the number after the equal plus spaces...
end

%% total number of slices:                
Nz = sliceArraySize * slicesPerSlab;

