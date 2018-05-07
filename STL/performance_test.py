from test import*
import cProfile
import pstats

if __name__ == "__main__":
	cProfile.run('run_program()', filename = "performance_result.stats")
	p=pstats.Stats('performance_result.stats')
	
	# Sorted by cumulative time
	p.sort_stats('cumulative')
	
	# Show the top 15%
	print('-'*50)
	p.print_stats(.15)