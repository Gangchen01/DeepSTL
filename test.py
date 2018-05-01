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
	Robust = Robustness(sys.argv, OPTION())
	# Create tree
	Robust.BiTree()
	# Create trajectory
	tree = Robust.tree


	struc = []
	data =[]

	EncodeSuccint(tree, struc, data)
	name =['x1']


	formuals = Formula(struc,data,name,5)

	vector = formuals.state_vector()

	act =np.array([3,0,5,0,0,0,1.2,0,1,2,3])

	tree1, method =formuals.get_action_tree(act)

	trr = formuals.combine_formula(tree1,tree,method)
	print_tree_indented(trr)

	trer = formuals.get_tree()

	tree2 = formuals.get_state_tree(tree1,method)

	formuals.update_state(trr)



	print(method)

	print(vector)










	#time = np.linspace(0,10,1001)
	#signal = np.array([time])
	#name = ['x1']
	#system = STL_Sys(name,signal,time)
	# Calculate robustness
	#Robust.Eval_Robust(system)
	

if __name__ == '__main__':
	run_program()
