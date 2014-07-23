function nfd = fclose(nfd)
%
% nfd = fclose(nfd)
% 
% Close the file and update handler's info
%

  if nfd.fileID > 0
    niftiMatlabIO('CLOSE', nfd.fileID);
  end
  nfd.fileID = int32(0);
  nfd.filePos = int32(0);
  nfd.hdrProcessed = false;
  
