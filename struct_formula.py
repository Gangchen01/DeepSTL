#!/usr/local/bin/python3

from binary_tree import *
from fractions import Fraction
import numpy as np
import random
from cal_robustness import *

class Formula:
    def __init__(self, struc =None,data =None, name= None, width = None):

        self.struc = struc
        self.data = data
        time = []
        for index in range(len(data)):
            if data[index]['Bound'] is None:
                time = np.append(time, np.array([0,0]))
            else:
                time = np.append(time, data[index]['Bound'])

        scale = []
        for index in range(len(data)):
            if data[index]['Value'][1] in ['>','>=','<','<=']:
                scale = np.append(scale,data[index]['Value'][2])
            else:
                scale = np.append(scale,np.array([0]))

        name_value= []
        for index in range(len(data)):
            if data[index]['Value'][0] in name:
                name_value = np.append(name_value, name.index(data[index]['Value'][0]))
            else:
                name_value = np.append(name_value, np.array([-1]))
        dir =[]
        for index in range(len(data)):
            if data[index]['Value'][1] in ['>','>=']:
                dir = np.append(dir,np.array([1]))
            elif data[index]['Value'][1] in ['<','<=']:
                dir = np.append(dir, np.array([-1]))
            else:
                dir = np.append(dir,np.array([0]))

        self.time = time
        self.scale = scale
        self.name_value = name_value
        self.dir =dir
        self.name = name
        self.width = width


    def get_action_tree(self,vector):
        #vector has 11 element [0-4 ] struc for tree, [5-6] time param, [7-9] predicate if applicable, [10] method
        method = vector[-1]
        if vector[2]>0:
            if vector[0] == 3:
                cargo = {'Value': 'alw', 'Bound':[vector[5], vector[6]+vector[5]]}
                name1 = self.name[int(vector[7])]
                dir =['<=', '>', '>=','<']
                right = Tree({'Value':[name1, dir[int(vector[8])], vector[9] ], 'Bound': None})
                tree = Tree(cargo)
                tree.right = right
            elif vector[0] ==4:
                cargo = {'Value': 'ev', 'Bound':[vector[5], vector[6]+vector[5]]}
                name1 = self.name[int(vector[7])]
                dir =['<=', '>', '>=','<']
                right = Tree({'Value':[name1, dir[int(vector[8])], vector[9]], 'Bound': None})
                tree = Tree(cargo)
                tree.right = right
            else:
                print('Invalid vector')
                return None
        elif vector[2] == 0:
            if vector[0] == 3:
                cargo = {'Value': 'alw', 'Bound':[vector[5], vector[6]]}
                tree = Tree(cargo)
            elif vector[0] ==4:
                cargo = {'Value': 'ev', 'Bound':[vector[5], vector[6]]}
                tree = Tree(cargo)
            else:
                print('Invalid vector')
                return None
        else:
            print('Invalid vector')
            return None

        return tree, method

    def combine_formula(self, tree_pre, tree_post, method):

        if method ==1:
            cargo ={'Value': 'and', 'Bound': None}
            tree = Tree(cargo)
            tree.left = tree_pre
            tree.right = tree_post
            return tree


        elif method == 2:
            cargo ={'Value': 'or', 'Bound': None}
            tree = Tree(cargo)
            tree.left = tree_pre
            tree.right = tree_post
            return tree
        elif method == 3 or method == 4:
            tree = tree_pre
            tree.right = tree_post
            return tree
        elif method == 0:
            return tree_pre
        # initialize
        else:
            print('Invalid method')

    def get_newstate_tree(self,action, method):
        return self.combine_formula(action,self.get_tree(),method)

    def get_tree(self):
        return DecodeSuccinct(self.struc, self.data)
    def update_state(self, tree):
        struc=[]
        data =[]
        EncodeSuccint(tree,struc,data)
        self.__init__(struc,data,self.name,self.width)

    def state_vector(self):
        zero = np.zeros(self.width - len(self.data))
        time = np.concatenate((self.time,  zero,zero), axis=0)
        scale =np.concatenate((self.scale,zero), axis=0)
        name_value = np.concatenate((self.name_value,zero),axis =0)
        dir =np.concatenate((self.dir, zero), axis =0)
        zero_struc =np.zeros(2*self.width+1 - len(self.struc))
        struct =np.concatenate((zero_struc,self.struc, ), axis=0)
        return np.concatenate((struct,time,scale,name_value,dir), axis =0)
    def model(self, act_vector):
        tree1, method = self.get_action_tree(act_vector)
        tree2 =self.get_tree()
        tree = self.combine_formula(tree1,tree2,method)
        self.update_state(tree)
        return self.state_vector()


def Init_state(name,signal,time):
    tf = time[-1]
    len_name= len(name)
    up = np.max(np.amax(signal,axis=0))
    lb = np.min(np.amin(signal,axis=0))

    bit1 = random.randint(3, 4)
    bit2 = 0
    bit3 = 5
    bit4 = 0
    bit5 = 0
    bit6 = random.random() * tf
    bit7 = random.random() * (tf - bit6)
    bit8 = random.randint(0, len_name - 1)
    bit9 = random.randint(0,3)
    bit10 = random.uniform(lb,up)
    bit11 = 0
    act = [bit1, bit2, bit3, bit4, bit5,bit6, bit7, bit8, bit9, bit10, bit11]
    formulas = Formula([],[],name,5)
    tree1, method = formulas.get_action_tree(act)
    tree2 = formulas.get_tree()
    tree = formulas.combine_formula(tree1, tree2, method)
    formulas.update_state(tree)

    return formulas.state_vector(), tf, len_name, up,lb, formulas

def robust(tree,name, signal, time):
    system = STL_Sys(name, signal, time)
    Robust = Robustness(sys.argv)
    Robust.SetTree(tree)
    return Robust.Eval(system)

def reward(tree,name, signalset, time):
    ls = np.size(signalset,0)
    robustness=[]
    for index in range(ls):
        sig = signalset[index]
        robustness= np.append(robustness,robust(tree,name,sig,time)[0])

    return min(robustness)





























