import subprocess, os

# vols = subprocess.Popen('ls ', shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[:-1]
cds = ['CD 1', 'CD 2', 'CD 3', 'CD 4', 'CD 5', 'CD 6']
for v in cds:
	os.chdir(v)
	os.system('for f in *.flac; do ffmpeg -i "$f" -acodec alac "${f%.flac}.m4a";  done')
	os.chdir('..')

for v in vols:
	os.chdir(v)
	for c in cds:
		os.chdir(c)
		os.system('rm -rf split-track*.flac')
		os.chdir('..')
	os.chdir('..')
		