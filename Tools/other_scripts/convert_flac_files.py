for l in lst:
	os.chdir(l)
	cue_file = subprocess.Popen('ls *.cue', shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[0]
	flac_file = subprocess.Popen('ls *.flac', shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[0]
	
	print cue_file, flac_file
	os.system('cuebreakpoints "%s" | shnsplit -o flac "%s"' % (cue_file, flac_file))
	os.system('cuetag "%s" split-track*.flac' % (cue_file) )
	os.system('for f in split-track*.flac; do ffmpeg -i "$f" -acodec alac "${f%.flac}.m4a";  done')
	os.chdir('..')