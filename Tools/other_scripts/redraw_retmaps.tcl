# this file to be read after running something like 
# tksurfer JS_071026js rh inflated -curv -tcl scriptStartLHAll.tcl

set hemiSphere "---HEMI---"
set condition "---CONDITION---"
set name = "---NAME---"

if {$hemiSphere == "rh"} {
	# for right hemisphere we need the following parameters
	set yDirection = -1
} else {		
# for left hemisphere we need the following parameters
	set yDirection = 1
}
set nrimages 72

set rgbname polar
set fthresh 2.3
set fslope 10
set fmid 1.4
set angle_offset 0.225
set angle_cycles 2.0
set invphaseflag 0
set revphaseflag 0
set smoothsteps 5


# run the standard script, taken from polar-views.tcl

#### file defaults: can reset in csh script with setenv
#set dir 004/image7                 ;# usually set in calling script setenv
set floatstem sig                   ;# float file stem
set realname 2                      ;# analyse infix
set complexname 3                   ;# analyse infix
set rgbname polar                   ;# name of rgbfiles

#### parm defaults: can reset in csh script with setenv
puts "tksurfer: [file tail $script]: read and smooth complex Fourier comp"
set overlayflag 1       ;# overlay data on gray brain
set surfcolor 1         ;# draw the curvature under data
set avgflag 1           ;# make half convex/concave
set complexvalflag 1    ;# two-component data
set colscale 0          ;# 0=wheel,1=heat,2=BR,3=BGR,4=twocondGR,5=gray
set angle_offset -.25   ;# phase offset (-0.25 for up semicircle start)
set angle_cycles 2.0    ;# adjust range
set fthresh 0.3         ;# val/curv sigmoid zero (neg=>0)
set fslope 1.5          ;# contast (was fsquash 2.5)
set fmid   0.8          ;# set linear region
set smoothsteps 10
set offset 0.20    ;# default lighting offset 

#### read non-cap setenv vars (or ext w/correct rgbname) to override defaults
source $env(FREESURFER_HOME)/lib/tcl/readenv.tcl

### for backward compatibility (old script-specific mechanism)
if { [info exists revpolarflag] } { 
  set revphaseflag $revpolarflag 
}

#### read curvature (or sulc)
puts "tksurfer: [file tail $script]: read curvature"
read_binary_curv

#### setenv polardir overrides setenv dir
if [info exists polardir] { set dir $polardir }

#### read and smooth complex component MRI Fourier transform of data
puts "tksurfer: [file tail $script]: read and smooth complex Fourier comp"
setfile val */$dir/${floatstem}${complexname}-$hemi.w     ;# polarangle
read_binary_values
smooth_val $smoothsteps 
shift_values     ;# shift complex component out of way

#### read and smooth real component MRI Fourier transform of data
puts "tksurfer: [file tail $script]: read and smooth real Fourier comp"
setfile val */$dir/${floatstem}${realname}-$hemi.w     ;# polarangle
read_binary_values
smooth_val $smoothsteps

#### scale and position brain
puts "tksurfer: [file tail $script]: scale, position brain"
open_window
make_lateral_view       ;# rotate either hemisphere
do_lighting_model -1 -1 -1 -1 $offset ;# -1 => nochange; diffuse curv (def=0.15)

# done with the standard script - here's my coding...


# smooth the curvature and surface before doing anything else
smooth_curv 40
shrink 200

rotate_brain_y 90 * yDirection

set presentY 0.0
for {set i 0} {$i < $nrimages} {incr i} {
	set l [string length "$i"] 
	if {$l == 1} {
	    set label "00$i"
	} elseif {$l == 2} {
	    set label "0$i"
	} elseif {$l == 3} {
	    set label "$i"
	}
    set fN [format "---FIGPATH---/%s_%s_%s_%s.tiff" $name $condition $hemiSphere $label]
	rotate_brain_y [yDirection * sin(6.283185307 * $l / $nrimages )]
	rotate_brain_x [cos(6.283185307 * $l / $nrimages )]
	redraw
	save_tiff $fN
	rotate_brain_y [yDirection * -sin(6.283185307 * $l / $nrimages )]
	rotate_brain_x [-cos(6.283185307 * $l / $nrimages )]
}

#exit