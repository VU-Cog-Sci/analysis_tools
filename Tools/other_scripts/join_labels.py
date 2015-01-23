#!/usr/bin/env python
# encoding: utf-8

import os, sys
from subprocess import *


# areas = [['V2',['V2d','V2v']],['V3',['V3d','V3v']],['V3AB',['V3A','V3B']],['LO',['LO1','LO2']],['VO',['VO1','VO2']]]
areas = [['IPS',['IPS1','IPS2','IPS3','IPS4']]]#,['V3',['V3d','V3v']],['V3AB',['V3A','V3B']],['LO',['LO1','LO2']],['VO',['VO1','VO2']]]
directory = 'retmap_PRF'
FSsubjects = ['BM_250913','DE_110412','EK_071014','JVS_091014','TK_091009tk','TN_081014','JW_310312']

for sj in FSsubjects:
	os.chdir(os.path.join(os.environ['SUBJECTS_DIR'], sj, 'label', directory ))
	print sj
	for i, area in enumerate(areas):
		print area[0]
		for hemi in ['lh','rh']:
			cmd = 'mri_mergelabels'
			cmd += ' -i ' + hemi + '.' + area[1][0] + '.label'
			cmd += ' -i ' + hemi + '.' + area[1][1] + '.label'
			cmd += ' -i ' + hemi + '.' + area[1][2] + '.label'
			cmd += ' -i ' + hemi + '.' + area[1][3] + '.label'
			cmd += ' -o ' + hemi + '.' + area[0] + '.label'
			os.system(cmd)

# for sj in FSsubjects:
# 	os.chdir(os.path.join(os.environ['SUBJECTS_DIR'], sj, 'label', directory ))
# 	for hemi in ['lh','rh']:
# 		os.system('cp %s.V3ab.label %s.V3AB.label'%(hemi, hemi))