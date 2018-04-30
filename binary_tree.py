#!/usr/local/bin/python3

# Define the Tree and Node
class Tree:
	def __init__(self, cargo, left=None, right=None):
		self.cargo = cargo
		self.left = left
		self.right = right

	def __str__(self):
		return str(self.cargo)

	def EncodeSuccint(self, struc, data):
		# If root is None , put 0 in structure array and return
		if self.cargo is None:
			struc.append(0)
			return

		# Else place 1 in structure array, key in 'data' array
		# and recur for left and right children
		struc.append(1)
		data.append(self.cargo)
		self.EncodeSuccint(self.left, struc, data)
		self.EncodeSuccint(self.right, struc, data)

	# Constructs tree from 'struc' and 'data'
	def DecodeSuccinct(self, struc, data):
		if (len(struc) <= 0):
			return None

		# Remove one item from structure list
		b = struc[0]
		struc.pop(0)

		# If removed bit is 1
		if b == 1:
			cargo = data[0]
			data.pop(0)

			# Create a tree node with removed data
			self.cargo = cargo

			# And recur to create left and right subtrees
			self.left = self.DecodeSuccinct(struc, data)
			self.right = self.DecodeSuccinct(struc, data)


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