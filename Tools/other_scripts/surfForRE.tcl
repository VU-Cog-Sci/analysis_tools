
# this file to be read after running something like 
# tksurfer JS_071026js rh inflated -curv -tcl scriptStartLHAll.tcl

set hemiSphere "---HEMI---"
set experiment "---EXPT---"

if {$hemiSphere == "rh"} {
	# for right hemisphere we need the following parameters
	set xRot 0.0
	set yRot 0.0
	set xTrans 0.0
	set yTrans 0.0
	set yStepSize 5.0
	set nrimages 72
} else {		
# for left hemisphere we need the following parameters
	set xRot 0.0
	set yRot 0.0
	set xTrans 0.0
	set yTrans 0.0
	set yStepSize 5.0
	set nrimages 72
}

set fthresh 0.1
set fmid 2.3
set fslope 0.5
set scale 0.85

# smooth the curvature and surface before doing anything else
smooth_curv 40
shrink 100

foreach ROI { T I P PgtI R1 R2 } {
	# read new overlay file
	set val [format "---SURFPATH---/%s-%s.w" $ROI $hemiSphere]
	sclv_read_from_dotw 0
	# setup overlay characteristics
	set gaLinkedVars(fthresh) $fthresh
	SendLinkedVarGroup overlay
	set gaLinkedVars(fmid) $fmid
	SendLinkedVarGroup overlay
	set gaLinkedVars(fslope) $fslope
	SendLinkedVarGroup overlay
	sclv_smooth 10 0
	
	# set the view back to its original position
	make_lateral_view
	scale_brain $scale

	translate_brain_x $xTrans
	translate_brain_y $yTrans
#	set colscalebarflag 1

#	rotate_brain_y $yRot
	rotate_brain_x $xRot
	
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
	    set fN [format "---FIGPATH---/%s_%s_%s_%s.tiff" $experiment $ROI $hemiSphere $label]
		rotate_brain_y $yStepSize
		redraw
		save_tiff $fN
		set presentY [expr { $presentY + $yStepSize }]
	} 
}

exit