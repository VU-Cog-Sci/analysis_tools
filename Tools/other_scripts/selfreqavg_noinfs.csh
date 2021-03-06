#! /bin/csh -f

#
# selfreqavg
#
# fMRI selective frequency averaging (for phase-encoding analysis)
#
# Original Author: Doug Greve
# CVS Revision Info:
#    $Author: greve $
#    $Date: 2007/11/28 00:01:02 $
#    $Revision: 1.10.2.2 $
#
# Copyright (C) 2002-2007,
# The General Hospital Corporation (Boston, MA). 
# All rights reserved.
#
# Distribution, usage and copying of this software is covered under the
# terms found in the License Agreement file named 'COPYING' found in the
# FreeSurfer source code root directory, and duplicated here:
# https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferOpenSourceLicense
#
# General inquiries: freesurfer@nmr.mgh.harvard.edu
# Bug reports: analysis-bugs@nmr.mgh.harvard.edu
#


set VERSION = '$Id: selfreqavg,v 1.10.2.2 2007/11/28 00:01:02 greve Exp $';
set cmdargv = "$argv";

set n = `echo $argv | grep version | wc -l` 
if($n != 0) then
  echo $VERSION
  exit 0;
endif

if($#argv == 0) goto usage_exit;

set instems = ();
set bext = "bshort";
set outstem = ();
set monly = 0;
set SynthSeed = 0;

set TR = ();

set stimtype   = ();
set ncycles    = ();
set direction  = ();
set directionlist  = (); # List of +/- ones #

set firstslice = ();
set detrend  = 1;
set nslices = ();
set rescale_target = 0;
set delay      = 0;
set slice_delay_file = ();
set nskip = 0;
set parname = ();
set hanrad = 0;
set fwhm = 0;
set cfgfile = ();

set usecfg = `echo $argv | egrep -e "-cfg" | wc -l`;
if($usecfg > 0) then
  goto parse_cfg;
  parse_cfg_return:
endif

goto parse_args;
parse_args_return:

goto check_params;
check_params_return:

#set MATLAB = `which octave`; #`getmatlab`;
set MATLAB = "octave --path $FSFAST_HOME/toolbox/:$FREESURFER_HOME/matlab/ "
# set MTLAB = "matlab"

if($status) exit 1;

if ($monly) then
  set QuitOnError = 0;
  set TARGET = "tee $mfile"
  rm -f $mfile;
else
  set QuitOnError = 1;
  set TARGET =  "$MATLAB " #"$MATLAB -display iconic"
endif  

set outdir = `dirname $outstem`;
mkdir -p $outdir
mkdir -p $outdir/omnibus
set foutstem = $outdir/omnibus/f;
set fsigstem = $outdir/omnibus/fsig;

#---------------------------------------------------------------#
$TARGET <<EOF

  global QuitOnError;
  QuitOnError = $QuitOnError;

  instems    = splitstring('$instems');
  bext       = '$bext';
  outstem    = '$outstem';
  foutstem   = '$foutstem';
  fsigstem   = '$fsigstem';
  slice_delay_file = deblank('$slice_delay_file');
  hanrad     = $hanrad;
  fwhm       = $fwhm;
  SynthSeed  = [$SynthSeed];
  inplaneres = 0;

  ext = getenv('FSF_OUTPUT_FORMAT');
  if(isempty(ext)) ext = 'bhdr'; end
  fprintf('Extension format = %s\n',ext);

  if(SynthSeed < 0) SynthSeed = sum(100*clock); end
  if(SynthSeed > 0)
    fprintf('Synthesizing: SynthSeed = %10d\n',SynthSeed);
    randn('state',SynthSeed); 
  end

  if(fwhm > 0)
    hanrad = pi*fwhm/(2*inplaneres*acos(.5));
  end

  fprintf('hanrad = %g\n',hanrad);

  sfa = fmri_sfastruct;
  sfa.analysistype = 'average';

  sfa.TR         = $TR;
  sfa.detrend    = $detrend;
  sfa.ncycles    = $ncycles;
  sfa.direction  = [$directionlist];
  sfa.stimtype   = '$stimtype';
  sfa.delay      = $delay;
  sfa.nskip      = $nskip;
  sfa.rescale_target = $rescale_target;
  sfa.infiles = instems;

  mri0 = MRIread(deblank(instems(1,:)),1);
  sfa.nrows = mri0.volsize(1);
  sfa.ncols = mri0.volsize(2);
  sfa.ntp   = mri0.nframes;
  firstslice = 0;
  lastslice = mri0.volsize(3)-1;
  sfa.firstslice = firstslice;
  sfa.lastslice = lastslice;
  mri0.vol = zeros([mri0.volsize 14]);

  mriF = mri0;
  mriF.vol = zeros([mri0.volsize 1]);
  mriFsig = mriF;

  %% Load the slice delay file %%
  if( ~isempty(slice_delay_file) )
    [slcid sfa.slice_delay] = fmri_ldslicedelay(slice_delay_file);
  else
    slcid = [];
    sfa.slice_delay = [];
  end

  %% Compute some basic numbers %%%
  sfa.Trun = sfa.TR * sfa.ntp;
  sfa.fundamental = sfa.ncycles/sfa.Trun;
  Nv = sfa.nrows * sfa.ncols;

  %% Account for skipping %%
  sfa.ntp = sfa.ntp - sfa.nskip;
  sfa.delay = sfa.delay - sfa.TR*sfa.nskip;

  %% Get the fft index of the fundamental %%
  ifund = fmri_indfreqfft(sfa.fundamental,sfa.ntp,sfa.TR);
  sfa.isignal = ifund;

  %% Compute the fft indices of the harmonics (excl fund) %%%
  iharm = [ifund+sfa.ncycles:sfa.ncycles:floor(sfa.ntp/2)];
  iharm = iharm(1:2); % Compatible with analyse.c %

  %% Get the fft indicies of the noise, exclude first 3 and +/- harm %%
  tmp = ones(floor(sfa.ntp/2),1);
  tmp(1:3)   = 0;
  tmp(ifund)   = 0;
  tmp(ifund-1) = 0;
  tmp(ifund+1) = 0;
  tmp(iharm)   = 0;
  tmp(iharm-1) = 0;
  tmp(iharm+1) = 0;
  sfa.inoise = find(tmp==1);

  %% Number of runs %%
  nruns = size(instems,1);

  %% Compute the degrees of freedom %%
  sfa.dof = (2 * length(sfa.inoise) - sfa.detrend) * nruns;

  %% phase of the delay %%
  ph_delay = sfa.delay * (2*pi)* sfa.fundamental;

  %%%%--------- Slice Loop ----------------%%%
  fprintf('Slice ');
  for slice = firstslice:lastslice
    fprintf('%3d ',slice);
    if(rem(slice,5)==19) fprintf('\n'); end

    sum_real_signal = 0;
    sum_imag_signal = 0;
    sum_noise_var   = 0;
    sum_wtrend      = 0;
    sum_fft0        = 0;

    for n = 1:nruns,
      instem = deblank(instems(n,:));
      %fprintf('Input Volume %s\n',instem);

      mri = MRIread(instem);
      if(isempty(mri))
         msg = sprintf('Could not load %s',instem);
         qoe(msg); error(msg);
      end
      f = squeeze(mri.vol(:,:,slice+1,:));
      if(SynthSeed > 0) f = randn(size(f)); end

      %%% Global Rescaling %%%
      if(sfa.rescale_target ~= 0) 
        fname = sprintf('%s.meanval',instem);
        fid = fopen(fname,'r');
        if(fid == -1) 
          msg = sprintf('Could not open %s',fname);
          qoe(msg); error(msg);
        end
        sfa.meanval = fscanf(fid,'%f');
        fclose(fid);
        sfa.rescale_factor = sfa.rescale_target / sfa.meanval;
        f = (f * sfa.rescale_factor);
      end

      %% Spatial Smoothing %%
      if(hanrad > 1)
        fprintf('Spatial Smoothing ...\n');
        HanFilter = fmri_hankernel(hanrad);
        f = fmri_spatfilter(f,HanFilter);
      end

      %% Skip if needed (not recommended) %%
      if(sfa.nskip > 0) 
        f = f(:,:,sfa.nskip+1:size(f,3));
      end

      %% Detrend (also removes mean) %%
      if(sfa.detrend > 0)   
        %fprintf('Detrending ...\n');
        [f wtrend] = fmri_detrend(f,[],2,zeros(sfa.ntp,1));
        %fprintf('Mean Offset = %g\n',mean(reshape1d(wtrend(:,:,1))));
        %fprintf('Mean Trend  = %g\n',mean(reshape1d(wtrend(:,:,2))));
        wtrend = reshape(wtrend, [Nv 2])'; %'
      else
        wtrend = zeros(2, Nv);
      end

      f = reshape(f, [Nv sfa.ntp])'; %'

      %% Compute fft, and mag/phase %%
      % Use conj so as to report phases as positive %
      f_fft   = conj(fft(f));
      mag_fft = abs(f_fft);
      phz_fft = angle(f_fft);

      %% Slice acquisition delay %%
      if( ~isempty(slcid) )
        ind = find(slcid == slice);
        if(isempty(ind))
          msg = sprintf('Slice %d not found in %s',slice,slice_delay_file);
          qoe(msg); error(msg);
        end
        slice_delay = sfa.slice_delay(ind);
        ph_slice_delay = slice_delay * (2*pi)* sfa.fundamental
      else
        slice_delay = 0;
        ph_slice_delay = 0;
      end

      %% Rotate by specified delay %%
      phz_fft_tmp = phz_fft - ph_delay - ph_slice_delay;

      %% Modify the phase for negative direction %%
      %if( strcmp(sfa.direction,'neg')) phz_fft = -phz_fft_tmp; 
      %else                             phz_fft =  phz_fft_tmp; 
      %end
      phz_fft = sfa.direction(n)*phz_fft_tmp; 

      %% Recompute the Real/Imag fft %%;
      f2_fft = mag_fft.*cos(phz_fft) + sqrt(-1)*mag_fft.*sin(phz_fft);
  
      %% Extract Real/Imag of Signal, use the rest to estimate noise %%
      sum_real_signal = sum_real_signal + real(f2_fft(ifund,:));
      sum_imag_signal = sum_imag_signal + imag(f2_fft(ifund,:));
      sum_noise_var   = sum_noise_var + mean(abs(f2_fft(sfa.inoise,:)).^2);
      sum_wtrend      = sum_wtrend + wtrend;
      sum_fft0        = sum_fft0 + real(f_fft(1,:));       

    end %% loop over number of runs %%

    real_signal = sum_real_signal/nruns;
    imag_signal = sum_imag_signal/nruns;
    var_noise   = sum_noise_var/nruns;
    ind0 = find(var_noise==0);
    var_noise(ind0) = 10^10;
    std_noise   = sqrt(var_noise);
    meanimg     = (sum_fft0 + sum_wtrend(1,:))/nruns; % one will be zero
    trendimg    = sum_wtrend(2,:)/nruns; 

    u = real_signal + sqrt(-1)*imag_signal;
    F = (abs(u).^2)./(var_noise/nruns); # jds: verified correct 8/6/09
    sigf = FTest(2,sfa.dof,F);
    log10sigf = -log10(sigf);

    % next two lines added by jds 1/7/2008:
    ceilind = find(isinf(log10sigf));
    log10sigf(ceilind) = 1000;

    phz = angle(u);
    t = imag(u)./sqrt(var_noise/nruns);

    %fprintf('Mean Phase = %g (%g)\n',mean(phz),mean(phz)*180/pi);

    %% Put into appropriate format and save %%
    tmp = zeros(13,Nv);
    tmp(1,:)  = log10sigf .* sign(t);
    tmp(2,:)  = log10sigf .* sin(phz);
    tmp(3,:)  = log10sigf .* cos(phz);
    tmp(4,:)  = F;
    tmp(5,:)  = sqrt(F) .* sin(phz);
    tmp(6,:)  = sqrt(F) .* cos(phz);
    tmp(7,:)  = std_noise;
    tmp(8,:)  = real_signal;
    tmp(9,:)  = imag_signal;
    tmp(10,:) = phz;
    tmp(11,:) = meanimg;
    tmp(12,:) = trendimg;
    tmp(13,:) = sqrt( real_signal.^2 + imag_signal.^2 );
    tmp(14,:) = phz .* (log10sigf > 2); % phase masked by sig .01

    ntmp  = size(tmp,1);
    tmp   = reshape(tmp', [sfa.nrows sfa.ncols 1 ntmp]); %'
    mri0.vol(:,:,slice+1,:) = tmp;

    tmp = reshape(F,[sfa.nrows sfa.ncols]);
    mriF.vol(:,:,slice+1) = tmp;

    tmp = reshape(log10sigf,[sfa.nrows sfa.ncols]);
    mriFsig.vol(:,:,slice+1) = tmp;

    %fname = sprintf('%s_%03d.bfloat',outstem,slice);
    %fmri_svbfile(tmp,fname);
    %fname = sprintf('%s_%03d.bfloat',foutstem,slice);
    %fmri_svbfile(tmp,fname);
    %fname = sprintf('%s_%03d.bfloat',fsigstem,slice);
    %fmri_svbfile(tmp,fname);

  end %% loop over slices %%
  fprintf('\n');

  fname = sprintf('%s.%s',outstem,ext);
  MRIwrite(mri0,fname);

  fname = sprintf('%s.%s',foutstem,ext);
  MRIwrite(mriF,fname);

  fname = sprintf('%s.%s',fsigstem,ext);
  MRIwrite(mriFsig,fname);


  %% Save the data file %%
  fprintf('Saving header to %s\n',outstem);
  fmri_svsfa(sfa,outstem);

  if(QuitOnError)  quit; end
  
EOF

echo "selfreqavg COMPLETED"

exit 0;

############--------------##################
parse_args:

set cmdline = "$argv";
while( $#argv != 0 )

  set flag = $argv[1]; shift;
  
  switch($flag)


    case "-i":
      if ( $#argv == 0) goto arg1err;
      set instems = ($instems $argv[1]); shift;
      breaksw

    case "-TR":
      if ( $#argv == 0) goto arg1err;
      set TR = $argv[1]; shift;
      breaksw

    case "-ncycles":
      if ( $#argv == 0) goto arg1err;
      set ncycles = $argv[1]; shift;
      breaksw

    case "-stimtype":
      if ( $#argv == 0) goto arg1err;
      set stimtype = $argv[1]; shift;
      breaksw

    case "-nskip":
      if ( $#argv == 0) goto arg1err;
      set nskip = $argv[1]; shift;
      breaksw

    case "-delay":
      if ( $#argv == 0) goto arg1err;
      set delay = $argv[1]; shift;
      breaksw

    case "-synth":
      if ( $#argv == 0) goto arg1err;
      set SynthSeed = $argv[1]; shift;
      breaksw

    case "-sdf":
      if ( $#argv == 0) goto arg1err;
      set slice_delay_file = $argv[1]; shift;
      if(! -e $slice_delay_file) then
        echo "ERROR: cannot find $slice_delay_file"
        exit 1;
      endif
      breaksw

    case "-direction":
      if ( $#argv == 0) goto arg1err;
      set direction = $argv[1]; shift;
      if($direction != pos && $direction != neg ) then
        echo "ERROR: direction must be either 'pos' or 'neg'"
        exit 1;
      endif
      if($direction == pos) set directionlist = ($directionlist +1);
      if($direction == neg) set directionlist = ($directionlist -1);
      breaksw

    case "-detrend":
      set detrend = 1;
      breaksw

    case "-nodetrend":
      set detrend = 0;
      breaksw

    case "-o":
      if ( $#argv == 0) goto arg1err;
      if ( $#outstem != 0 ) then
        echo ERROR: only one outstem allowed.
        exit 1
      endif
      set outstem = $argv[1]; shift;
      breaksw

    case "-firstslice":
    case "-fs":
      if ( $#argv == 0) goto arg1err;
      set firstslice = $argv[1]; shift;
      breaksw

    case "-nslices":
    case "-ns":
      if ( $#argv == 0) goto arg1err;
      set nslices = $argv[1]; shift;
      breaksw

    case "-rescale":
      if ( $#argv == 0) goto arg1err;
      set rescale_target = $argv[1]; shift;
      breaksw

    case "-hanrad":
      if ( $#argv == 0) goto arg1err;
      set hanrad = $argv[1]; shift;
      breaksw

    case "-fwhm":
      if ( $#argv == 0) goto arg1err;
      set fwhm = $argv[1]; shift;
      breaksw

    case "-monly":
      if ( $#argv == 0) goto arg1err;
      set mfile = $argv[1]; shift;
      set monly = 1; 
      breaksw

    case "-parname":
      if ( $#argv == 0) goto arg1err;
      set parname = $argv[1]; shift;
      breaksw

    case "-baseline":
      if ( $#argv == 0) goto arg1err;
      # automatic -- ignore #
      breaksw

    case "-cfg":
      if ( $#argv == 0) goto arg1err;
      set cfgfile = $argv[1]; shift;
      if(! -e $cfgfile ) then
        echo "ERROR: $cfgfile does not exist"
        exit 1;
      endif
      breaksw

    # Ignore
    case "-noautostimdur":
      breaksw

    # Ignore
    case "-acfbins":
      shift;
      breaksw

    case "-debug":
      set verbose = 1;
      breaksw

    case "-echo":
      set echo = 1;
      breaksw

    case "-debug":
      set echo = 1;
      set verbose = 1;
      breaksw

    case "-umask":
      if ( $#argv == 0) goto arg1err;
      set umaskarg = "-umask $argv[1]";
      umask $argv[1]; shift;
      breaksw

    default:
      echo ERROR: Flag $flag unrecognized.
      echo $cmdline
      exit 1
      breaksw
  endsw

end
goto parse_args_return;
############--------------##################

############--------------##################
check_params:

  if($#instems == 0) then
    echo "ERROR: must specify at least one input volume";
    exit 1;
  endif

  if($#outstem == 0) then
    echo "ERROR: must specify an output volume";
    exit 1;
  endif

  if($#TR == 0) then 
    echo "ERROR: must specify a TR";
    exit 1;
  endif

  ## Check that parname exists in each input directory ##
  ## If so, get ncycles and direction, unless spec on cmdline ##
  if($#parname != 0) then
    set directionlist = ();
    foreach instem ($instems)
      set indir = `dirname $instem`;
      set parfile = $indir/$parname;
      if(! -e $parfile) then
        echo "ERROR: cannot find $parfile"
        exit 1;
      endif
      if($#ncycles == 0) then
        set tmp = `grep ncycles $parfile`;
        if($#tmp == 2) then 
          set ncycles = $tmp[2];
        else
          echo "ERROR: ncycles not specified"
          exit 1;
        endif
      endif
      if($#direction == 0) then
        set tmp = `grep direction $parfile`;
        if($#tmp == 2) then
          set direction = $tmp[2];
        else
          set direction = pos;
        endif
	if($direction == pos) then
          set directionlist = ($directionlist +1);
        else
          set directionlist = ($directionlist -1);
        endif
        set direction = (); # Hack to keep getting direction from parfile
      endif
      if($#stimtype == 0) then
        set tmp = `grep stimtype $parfile`;
        if($#tmp == 2) then
          set stimtype = $tmp[2];
        endif
      endif
    end
  endif

  if($#direction == 0 && $#directionlist == 0) then
    set direction = "pos";
    set directionlist = ();
    foreach instem ($instems)
      set directionlist = ($directionlist +1);
    end
  endif
  echo "DirectionList: $directionlist"

  if($#ncycles == 0) then
    echo "ERROR: ncycles not set"
    exit 1;
  endif

  if($#stimtype == 0) then
    set stimtype = "unspecified";
  else
    if($stimtype != eccen && $stimtype != polar) then
      echo "ERROR: stimtype must be either 'eccen' or 'polar'"
      exit 1;
    endif
  endif
    
  if($hanrad != "0" && $fwhm != "0") then
    echo "ERROR: cannot specify hanrad and fwhm"
    exit 1;
  endif

goto check_params_return;
############--------------##################


############--------------##################
arg1err:
  echo "ERROR: flag $flag requires one argument"
  exit 1
############--------------##################


############--------------##################
parse_cfg:
  @ n = 1;
  while($n < $#argv)
    if("$argv[$n]" == "-cfg") then
      @ m = $n + 1;
      set cfgfile = $argv[$m];
      if(! -e $cfgfile ) then
        echo "ERROR: $cfgfile does not exist"
        exit 1;
      endif
    endif
    @ n = $n + 1;
  end

  if($#cfgfile == 0) then
    echo "ERROR: must specify config file with -cfg flag";
    exit 1;
  endif

  echo "--- Parsing Config File: $cfgfile ----"
  set cfgargs = `cat $cfgfile`;
  echo $cfgargs
  set argv = ($cfgargs $argv);

goto parse_cfg_return;
#--------------------------------------------------------------------#

############--------------##################
usage_exit:
  echo "USAGE: selfreqavg"
  echo "   -i instems  ...   : input stem(s)"
  echo "   -TR TR            : temporal resolution (sec)"
  echo "   -o outstem        : output stem"
  echo "   -parname name     : parfile, same in each instem directory"
  echo "   -stimtype  type   : eccen or polar (see also parname)"
  echo "   -ncycles ncycles  : number of stimulation cycles (see also parname)"
  echo "   -direction string : pos or neg (see also parname)"
  echo "   -delay   delay    : global delay (in seconds)"
  echo "   -sdf     file     : slice delay file "
  echo "   -nskip nskip      : skip the first nskip TRs"
  echo "   -nodetrend        : do not remove linear trend"
  echo "   -hanrad radius    : hanning radius for in-plane smoothing"
  echo "   -fwhm   width     : full-width/half-max (mm)"
  echo "   -ipr resolution   : in-plane resolution (mm)"
  echo "   -rescale target   : rescale so global mean equals target"
  echo "   -synth seed       : -1 for automatic, 0 for no synth (default)"
  echo "   -firstslice fs    : first anatomical slice"
  echo "   -nslices    ns    : number of anatomical slices"
  echo "   -monly mfile      : just create a matlab file"
  echo "   -umask umask      : set unix file permission mask"
  echo "   -version          : print version and exit"
exit 1;
