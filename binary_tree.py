#!/usr/local/bin/python3

# Define the Tree and Node
class Tree:
	def __init__(self, cargo, left=None, right=None):
		self.cargo = cargo
		self.left = left
		self.right = right

	def __str__(self):
		return str(self.cargo)

# Print the Tree formula
def print_tree(tree):
	if tree is None: return
	print_tree(tree.left)
	print(tree.cargo['Value'], end="")
	print_tree(tree.right)

# Print the Tree structure
# (Note: The most left node is the root, the most right nodes are the leaves.
# Downward is left, upward is right)
def print_tree_indented(tree, level=0):
	if tree is None: return
	print_tree_indented(tree.right, level+1)
	print("---|" * level + str(tree.cargo['Value']))
	print_tree_indented(tree.left, level+1)

# calculate the number of layers
def calculate_layers(tree, current=0):
	if tree is None: return current
	left  = calculate_layers(tree.left, current+1)
	right = calculate_layers(tree.right, current+1)
	return max(left, right)