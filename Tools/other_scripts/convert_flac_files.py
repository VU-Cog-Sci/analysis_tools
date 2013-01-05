import subprocess, os

vols = subprocess.Popen('ls ', shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[:-1]
cds = ['CD 1', 'CD 2', 'CD 3']
for v in vols:
	os.chdir(v)
	for c in cds:
		os.chdir(c)
		cue_file = 'CDImage.cue' # subprocess.Popen('ls *.cue', shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[0]
		flac_file = 'CDImage.flac' # subprocess.Popen('ls *.flac', shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[0]
	
		print cue_file, flac_file
		os.system('ffmpeg -i CDImage.ape CDImage.flac')
		os.system('cuebreakpoints "%s" | shnsplit -o flac "%s"' % (cue_file, flac_file))
		os.system('cuetag "%s" split-track*.flac' % (cue_file) )
		os.system('for f in split-track*.flac; do ffmpeg -i "$f" -acodec alac "${f%.flac}.m4a";  done')
		os.chdir('..')
	os.chdir('..')

for v in vols:
	os.chdir(v)
	for c in cds:
		os.chdir(c)
		os.system('rm -rf split-track*.flac')
		os.chdir('..')
	os.chdir('..')
		