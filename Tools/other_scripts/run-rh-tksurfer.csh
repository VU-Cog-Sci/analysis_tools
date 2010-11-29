#!/bin/csh -f

set name = DB_290910db
set hemi = rh

setenv eccendir eccen/11/surf
setenv polardir polar/5/surf

#setenv flatzrot 110
#setenv flatscale 1.4
setenv revphaseflag 0

setenv patchname occip.patch.flat
#setenv patchname full.patch.flat
#setenv patchname parietal.patch.flat
setenv offset 0.4

setenv rgbname eccen
setenv fthresh 1.0
setenv fslope 10
setenv fmid 1.05
setenv angle_offset 0.22
setenv angle_cycles 1.0
setenv invphaseflag 0
setenv smoothsteps 5 #from default
setenv revphaseflag 1


################-views is for viewing on inflated, -flat for occip patch
##########use inflated to make cuts using retino data

#tksurfer -$name $hemi inflated -tcl $FREESURFER_HOME/lib/tcl/eccen-flat.tcl
tksurfer -$name $hemi inflated -tcl $FREESURFER_HOME/lib/tcl/eccen-views.tcl

setenv rgbname polar
#setenv fthresh 0.4
#setenv fslope 2
#setenv fmid 0.8
setenv fthresh 1.3
setenv fslope 10
setenv fmid 1.4
setenv angle_offset 0.00
setenv angle_cycles 2.0
#setenv angle_cycles 1.0 # s6 scanned with double-wedge stim
setenv invphaseflag 0
setenv revphaseflag 0
setenv smoothsteps 5 #was 5

#tksurfer -$name $hemi inflated -tcl $FREESURFER_HOME/lib/tcl/polar-flat.tcl
tksurfer -$name $hemi inflated -tcl $FREESURFER_HOME/lib/tcl/polar-views.tcl

unsetenv angle_offset
unsetenv angle_cycle

#setenv patchname patch
setenv revpolarflag 0

setenv rgbname fs
setenv fthresh 0.3
setenv fslpoe 5.0
setenv fmid 0.8
setenv angle_offset 0.8
setenv revphaseflag 1

#tksurfer -$name $hemi inflated -tcl fs-make.tcl
#tksurfer -$name $hemi inflated -tcl fs-flat.tcl

Exit 0
