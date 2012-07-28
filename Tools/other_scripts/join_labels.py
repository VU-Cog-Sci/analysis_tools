#!/usr/bin/env python
# encoding: utf-8

import os, sys
from subprocess import *


areas = [['V2',['V2d','V2v']],['V3',['V3d','V3v']],['V3AB',['V3A','V3B']]]
directory = 'retmap'
FSsubjects = ['TD_230712_GPU']

for sj in FSsubjects:
	os.chdir(os.path.join(os.environ['SUBJECTS_DIR'], sj, 'label', directory ))
	print sj
	for i, area in enumerate(areas):
		print area[0]
		for hemi in ['lh','rh']:
			cmd = 'mri_mergelabels'
			cmd += ' -i ' + hemi + '.' + area[1][0] + '.label'
			cmd += ' -i ' + hemi + '.' + area[1][1] + '.label'
			cmd += ' -o ' + hemi + '.' + area[0] + '.label'
			os.system(cmd)
