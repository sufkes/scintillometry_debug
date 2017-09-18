# This program takes a time profile generated using cProfile, and sorts and prints the output for each module in order of decreasing cumulative time.

import sys
import pstats

if (len(sys.argv) != 2) and (len(sys.argv) != 3):
	print "Specify filename of profile [and number of lines to print]."
	sys.exit()

filename	= str(sys.argv[1])			# Specify filename.
if len(sys.argv) == 3:
	numlines = int(sys.argv[2])
else:
	numlines		= 50				# Lines to print

p = pstats.Stats(filename)				# Load profile.
p.strip_dirs()							# Strip long directory names.
p.sort_stats('cumulative').print_stats(numlines)	# Print output.
