#!/usr/local/bin/python3

from cal_robustness import *
from struct_formula import *
# import numpy as np

# Print-out control panel
class OPTION:
	SHOW_RAW        = False
	SHOW_LIST       = False
	SHOW_COMP_LIST  = False
	SHOW_TREE_FORM  = False
	SHOW_TREE_STRUC = True
	SHOW_STAT       = False
	SHOW_ROBUST		= True
	SHOW_TIME		= False

def run_program():
	# Read formula and state


	name =['x1']
	time1 =np.linspace(0,10,1001)
	#signal = np.array([time1])
	signal =[]
	sig= np.random.rand(1,1001)
	for index in range(100000):
		signal.append(sig)

	t0 = time()
	state, tf, len_name, up, lb, formula = Init_state(name, signal, time1)
	tree= formula.get_tree()
	#print_tree_indented(tree)
	t1 = time()
	param = (tree,name,signal[0],time1)
	rewards= reward(tree,name,signal,time1)
	t2 = time()
	rewardd = poolreward(tree,name,signal,time1)
	t3 = time()
	print(t2-t1)
	print(t3-t2)















	#time = np.linspace(0,10,1001)
	#signal = np.array([time])
	#name = ['x1']
	#system = STL_Sys(name,signal,time)
	# Calculate robustness
	#Robust.Eval_Robust(system)
	

if __name__ == '__main__':
	run_program()
