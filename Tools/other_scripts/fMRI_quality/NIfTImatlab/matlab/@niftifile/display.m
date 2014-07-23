function display(nfd)
% display(nfd) 
%
% Display a niftifile object

  format compact;
  display(nfd.niftiheader);
  if nfd.compressed
    fprintf('compression\t: ON\n');
  else
    fprintf('compression\t: OFF\n');
  end
  
  fprintf('status\t\t: ');
  if nfd.fileID == 0
    fprintf('closed\n');
  else
    fprintf('open\n');
  end
  
  if nfd.hdrProcessed
    fprintf('header\t\t: loaded\n');
  else
    fprintf('header\t\t: empty\n');
  end
  
  fprintf('\n');
