#!/usr/local/bin/python3

from binary_tree import *
from fractions import Fraction
import numpy as np

class Formula:
    def __init__(self, struc =None,data =None):

        self.struc = struc
        self.data = data
        time = []
        for index in range(len(data)):
            if data[index]['Bound'] is None:
                time = np.concatenate((time, np.array([0,0])), axis =0)
            else:
                time = np.concatenate((time, data[index]['Bound']), axis =0)

        scale = []
        for index in range(len(data)):
            if data[index]['Value'][1] in ['>','>=','<','<=']:
                scale = np.concatenate((scale,data[index]['Value'][2]),axis=0)
            else:
                scale = np.concatenate((scale,np.array([0])), axis =0)

        name= []
        for index in range(len(data)):
            if data[index]['Value'][1] in ['>','>=','<','<=']:
                name = np.concatenate((name,data[index]['Value'][0]),axis=0)
            else:
                name = np.concatenate((name,None), axis =0)
        dir =[]
        for index in range(len(data)):
            if data[index]['Value'][1] in ['>','>=']:
                dir = np.concatenate((dir,np.array([1])),axis=0)
            elif data[index]['Value'][1] in ['<','<=']:
                dir = np.concatenate((dir, np.array([-1])), axis=0)
            else:
                dir = np.concatenate((dir,np.array([0])), axis =0)

        self.time = time
        self.scale = scale
        self.name = name
        self.dir =dir
        self.width = len(data)
        self.structure = DecodeSuccinct(struc,data)

    def Init_actions(self,time_slices):
        formulas = list()
        return formulas

    def combine_formula(self, formula_pre, formula_post, method):

        if method ==1:
            cargo ={'Value': 'and', 'Bound': None}
            tree = Tree(cargo)
            tree.left = formula_pre
            tree.right = formula_post
            struc=[]
            data = []
            EncodeSuccint(tree,struc,data)
            self.__init__(struc,data)
        elif method == 2:
            cargo ={'Value': 'or', 'Bound': None}
            tree = Tree(cargo)
            tree.left = formula_pre
            tree.right = formula_post
            struc=[]
            data = []
            EncodeSuccint(tree,struc,data)
            self.__init__(struc,data)
        elif method == 3:
            tree = formula_pre
            tree.right = formula_post
            struc=[]
            data = []
            EncodeSuccint(tree,struc,data)
            self.__init__(struc,data)
        else:
            print('Invalid method')

    def state(self,action):
        formuas = action
        return formuas









