#!/usr/local/bin/python3

import sys
from reference import *

# Read the formula file. If the file does not exist, exit
# the program. If the file is empty, exit the program.
def read_file(filename):
	try:
		with open(filename) as f:
			str = f.read()
		if not str:
			sys.exit("\033[1;31;47m Error: File '%s' is empty. \033[0m" %filename)
	except FileNotFoundError as err:
		sys.exit("\033[1;31;47m %s \033[0m" %err)
	else:
		return str

# Read the state file. If the file does not exist, exit
# the program. If the file is empty, exit the program.
def read_state(filename):
	try:
		state = list()
		with open(filename) as f:
			for line in f:
				state.append((line.strip('\n')).strip(','))
		if not state:
			sys.exit("\033[1;31;47m Error: File '%s' is empty. \033[0m" %filename)
	except FileNotFoundError as err:
		sys.exit("\033[1;31;47m %s \033[0m" %err)
	else:
		return state

# Delete all the spaces and newline characters.
def preprocess(str):
	str = str.strip('\n')
	str = str.replace(' ','')
	return str

# Show the statistics of the formula. If show = True, print
# out the result. Check any potential syntax error in the formula
def stat(str, STATE, SHOW):
	number_ev                = str.count('ev_')
	number_alw               = str.count('alw_')
	number_not               = str.count('not_')
	number_until             = str.count('until_')
	number_and               = str.count('and_')
	number_or                = str.count('or_')
	number_open_parenthesis  = str.count('(')
	number_close_parenthesis = str.count(')')
	number_open_bracket      = str.count('[')
	number_close_bracket     = str.count(']')
	number_state             = [0]*len(STATE)
	for i in range(len(STATE)): number_state[i] = str.count(STATE[i])

	if sum(number_state) == 0:
		sys.exit("\033[1;31;47m Error: Missing states in the formula! \033[0m")
	if number_open_parenthesis != number_close_parenthesis:
		sys.exit("\033[1;31;47m SyntaxError: The number of parenthesis does not match! \033[0m")
	if number_open_bracket != number_close_bracket:
		sys.exit("\033[1;31;47m SyntaxError: The number of bracket does not match! \033[0m")
	if (number_ev + number_alw + number_until) != number_open_bracket:
		sys.exit("\033[1;31;47m SyntaxError: Missing bracket in the formula! \033[0m")
	if SHOW:
		print("-"*50)
		print("Number of Eventually:        ", number_ev)
		print("Number of Always:            ", number_alw)
		print("Number of Negation:          ", number_not)
		print("Number of Until:             ", number_until)
		print("Number of Conjunction:       ", number_and)
		print("Number of Disjunction:       ", number_or)
		print("Number of Open parenthesis:  ", number_open_parenthesis)
		print("Number of Close parenthesis: ", number_close_parenthesis)
		print("Number of Open bracket:      ", number_open_bracket)
		print("Number of Close bracket:     ", number_close_bracket)
		for i in range(len(STATE)): 
			print("Number of %s:                " %STATE[i], number_state[i])

# Check if any state exists in the string
def check_state(index, str, STATE):
	for x in STATE:
		if str[index:index+len(x)] == x:
			return x
	return None
	   
# Capture the character in formula
def get_formula(index, str, formula, STATE):
	if str[index:index+3] == 'ev_':
		formula.append('ev_')
		index = index + 3
		if str[index] != '[':
			sys.exit("\033[1;31;47m SyntaxError: Missing open bracket! \033[0m")
	elif str[index:index+4] == 'alw_':
		formula.append('alw_')
		index = index + 4
		if str[index] != '[':
			sys.exit("\033[1;31;47m SyntaxError: Missing open bracket! \033[0m")
	elif str[index:index+4] == 'not_':
		formula.append('not_')
		index = index + 4
		if str[index] != '(':
			sys.exit("\033[1;31;47m SyntaxError: Missing open parenthesis! \033[0m")
	elif str[index:index+6] == 'until_':
		formula.append('until_')
		index = index + 6
		if str[index] != '[':
			sys.exit("\033[1;31;47m SyntaxError: Missing open bracket! \033[0m")
	elif str[index:index+4] == 'and_':
		formula.append('and_')
		index = index + 4
		if str[index] != '(':
			sys.exit("\033[1;31;47m SyntaxError: Missing open parenthesis! \033[0m")
	elif str[index:index+3] == 'or_':
		formula.append('or_')
		index = index + 3
		if str[index] != '(':
			sys.exit("\033[1;31;47m SyntaxError: Missing open parenthesis! \033[0m")
	elif check_state(index, str, STATE):
		x = check_state(index, str, STATE)
		formula.append(x)
		index = index + len(x)
	elif str[index:index+4] in SPECIAL_CHAR_4:
		formula.append(str[index:index+4])
		index = index + 4 
	elif str[index:index+2] in SPECIAL_CHAR_2:
		formula.append(str[index:index+2])
		index = index + 2
	elif str[index] in SPECIAL_CHAR_1:
		if str[index] == '(' and index == len(str)-1:
			sys.exit("\033[1;31;47m SyntaxError: Improper use of parenthesis! \033[0m")
		elif str[index] == '[' and index == len(str)-1:
			sys.exit("\033[1;31;47m SyntaxError: Improper use of bracket! \033[0m")
		elif str[index] == '[':
			index_end = index + 1
			while str[index_end] != ']':
				if str[index_end] == '[':
					sys.exit("\033[1;31;47m SyntaxError: Improper use of bracket! \033[0m")
				else: index_end += 1
			if index_end - index < 4 or str[index:index_end+1].count(',') != 1:
				sys.exit("\033[1;31;47m SyntaxError: Improper use of bracket! \033[0m")
			else:
				formula.append(str[index:index_end+1])
				index =  index_end + 1
		else:
			formula.append(str[index])
			index += 1
	elif str[index] in ['-','+'] or str[index].isdigit():
		temp = str[index]
		index += 1
		while str[index].isdigit() or str[index] in ['.','/','*','+','-']:
			temp = temp + str[index]
			index += 1
		formula.append(temp)
	else:
		sys.exit("\033[1;31;47m SyntaxError: Unknown character: '%s' \033[0m" %str[index])
	return index, formula  
