% Script that executes pulse oximetry (PPU) 3T Philips logfile. Just press play (F5)
%
% 
% Note: 
% - This is the input script to the PhysIO toolbox. Only this file has to be adapted for your study.
% - For documentation of any of the defined substructures here, please
% see also tapas_physio_new.m or the Manual_PhysIO-file.
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: main_PPU.m 411 2014-01-20 02:40:51Z kasperla $
%
%% 0. Put code directory into path; for some options, SPM should also be in the path
pathRETROICORcode = fullfile(fileparts(mfilename('fullpath')), ...
    '../../../code');

addpath(genpath(pathRETROICORcode));

physio      = tapas_physio_new();
log_files   = physio.log_files;
thresh      = physio.thresh;
sqpar       = physio.sqpar;
model       = physio.model;
verbose     = physio.verbose;

%% 1. Define Input Files

log_files.vendor            = 'Philips';
log_files.cardiac           = '---LOG_FILE---';      
log_files.respiration       = '---LOG_FILE---';      


%% 2. Define Nominal Sequence Parameter (Scan Timing)

% 2.1. Counting scans and dummy volumes from end of run, i.e. logfile
sqpar.Nslices           = ---NR_SLICES---;
sqpar.NslicesPerBeat    = ---NR_SLICES_PER_BEAT---;
sqpar.TR                = ---TR_SECS---;
sqpar.Ndummies          = ---NR_DUMMIES---;
sqpar.Nscans            = ---NR_TRS---;
sqpar.onset_slice       = ---ONSET_SLICE---;

% 2.2. Counting scans and dummy volumes from beginning of run, i.e. logfile,
%      includes counting of preparation gradients        
% (Uncomment the following line to execute) 
% sqpar.Nprep = 3;


%% 3. Define Gradient Thresholds to Infer Gradient Timing (Philips only)
thresh.scan_timing = struct('zero', 700, 'slice', 1800, 'vol', [], ...
 'grad_direction', '---GRADIENT_DIRECTION---');
thresh.scan_timing.vol = [];
thresh.scan_timing.vol_spacing = []; % in seconds


%% 4. Define which Cardiac Data Shall be Used

%% 4.1. Using heart beat events logged prospectively during scanning instead
thresh.cardiac.modality = 'OXY'; % 'ECG','ECG_raw', 'OXY' or 'OXYGE' (for pulse oximetry)

%% 4.2. Using ECG time curve to detect heartbeat events, via a chosen or
%% saved reference R-peak
thresh.cardiac.initial_cpulse_select.min = 1;
thresh.cardiac.initial_cpulse_select.file = ''; % used to save reference peak or load it from there, if manual_peak_select == true
thresh.cardiac.initial_cpulse_select.method = 'manual'; % 'load_from_logfile', 'manual' or 'load' (from previous manual run)


%% 5. Order of RETROICOR-expansions for cardiac, respiratory and
%% interaction terms. Option to orthogonalise regressors

model.type = 'RETROICOR_HRV_RVT';
model.order = struct('c',3,'r',4,'cr',1, 'orthogonalise', 'none');
model.input_other_multiple_regressors = ''; % either txt-file or mat-file with variable R
model.output_multiple_regressors = '---OUTPUT_FILE_NAME---';


%% 6. Output Figures to be generated

verbose.level = 3;
% 0 = none; 
% 1 = main plots (default); 
% 2 = debugging plots: for missed slice/volume events, missed heartbeats, 1D time series of created regressors
% 3 = all plots, incl. cardiac/respiratory phase estimation,
%     slice-to-volume assignment
verbose.fig_output_file = 'PhysIO_output.ps';


%% 7. Run the main script with defined parameters

physio.log_files    = log_files;
physio.thresh       = thresh;
physio.sqpar        = sqpar;
physio.model        = model;
physio.verbose      = verbose;

% physio = tapas_physio_new('manual_peak_select', physio);
% physio.thresh.cardiac.posthoc_cpulse_select.method = 'load';

[physio_out, R, ons_secs] = tapas_physio_main_create_regressors(physio);
