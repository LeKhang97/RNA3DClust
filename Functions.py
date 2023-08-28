import numpy as np
import sys
import numpy as np
import itertools
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from numpyencoder import NumpyEncoder
import os
import json

def flatten(l):
    result = []
    for sublist in l:
        if isinstance(sublist, list):
            result += sublist
        else:
            result += [sublist]

    return result

def flatten_np(l):
    return np.asarray(np.concatenate(l))

def get_coordinate(x):
    xs = []
    ys = []
    zs = []
    for line in x:
        xs += [float(line[8])]
        ys += [float(line[9])]
        zs += [float(line[10])]
    
    return xs, ys, zs

def process_pdb(list_format, n = 3, models = True):
    coor_atoms_C = []
    chains = []
    res_num = []
    result = []
    l = [(0,6),(6,11),(12,16),(16,17),(17,20),(21,22),(22,26),
         (26,27),(30,37),(38,46),(46,54),(54,60),(60,66),(72,76),
          (76,78),(78,80)]
    model = ''
    num_model = 0
    for line in list_format:
        if 'MODEL' in line[:5]:
            num_model += 1
            if models == False and num_model > 1:
                break
            model = line.replace(' ','')


        if "ATOM" in line[:6].replace(" ","") and len(line[17:20].replace(" ","")) == 1 and f"C{n}" in line[12:16]:
            new_line = [line[v[0]:v[1]].replace(" ","") for v in l ] + [model]
            #print(new_line)
            
            if new_line[5] + '_' + model not in chains:
                chains += [new_line[5] + '_' + model]
            coor_atoms_C += [new_line]

    if bool(chains) == 0:
        return False
    
    for chain in chains:
        sub_coor_atoms_C = [new_line for new_line in coor_atoms_C if new_line[5] == chain.split('_')[0]
                            and new_line[16] == ''.join(chain.split('_')[1:])]
        #print(sub_coor_atoms_C)
        result += [get_coordinate(sub_coor_atoms_C)]
        res_num += [[int(new_line[6]) for new_line in coor_atoms_C if new_line[5] == chain.split('_')[0]
                            and new_line[16] == ''.join(chain.split('_')[1:])]]

    return result, chains, res_num

def list_to_range(l):
    l = list(set(l))
    l2 = []
    s = l[0]
    for p,v in enumerate(l):
        if p >= 1:
            if v == l[p-1] + 1:
                if p == len(l) - 1:
                    l2 += [range(s,v+1)]
                    
                continue
                
            e = l[p-1] + 1
            l2 += [range(s,e)]
            s = v
            
        if p == len(l) - 1:
            l2 += [range(s,v+1)]
    
    l2 = list(set(l2))

    l2 = sorted(l2, key=lambda x: x[0])
    
    return l2

def pymol_proccess(pred, res_num, name = None, color = None):
    if color == None:
        color = ['red', 'green', 'yellow', 'orange', 'blue', 'pink', 'cyan', 'purple', 'white', 'grey', 'brown']    

    label_set = list(set(pred))

    cmd = []
    for num, label in enumerate(label_set):
        label1 = [res_num[p] for p,v in enumerate(pred) if v == label]
        clust_name = name + f'_cluster_{num}' if name != None else f'cluster_{num}'
        cmd += [command_pymol(label1, clust_name, color[num])]

    return cmd

def command_pymol(l, name, color):
    l2 = list_to_range(l)
    mess = f'select {name}, res '
    for p,r in enumerate(l2):
        if len(r) > 1:
            mess += f'{r[0]}-{r[-1]+1}'
            if p != len(l2) - 1:
                mess += '+'
        else:
            mess += f'{r[0]}' + '+'
    mess += f'; color {color}, {name}'
    print(mess)
    return mess

def cluster_algo(*args):
    data = args[0]
    if args[1] == 'D':
        model = DBSCAN(eps=args[2], min_samples= args[3])
    elif args[1] == 'M':
        model = MeanShift(bandwidth= args[2])
    elif args[1] == 'A':
        model = AgglomerativeClustering(n_clusters= args[2])
    elif args[1] == 'S':
        model = SpectralClustering(n_clusters= args[2], gamma= args[3])
    else:
        print(args[1])
        sys.exit("Non recognized algorithm!")

    pred = model.fit_predict(data)


    return pred

def check_C(result, threshold):
    data = []
    if result == False or len(result) == 0:
        return False

    else:
        for t in range(len(result[0])):
            if len(result[0][t][0]) < threshold:
                continue
        
            l = [[result[0][t][0][i], result[0][t][1][i], result[0][t][2][i]] for i in range(len(result[0][t][0]))]
            data += [np.array(l)]

        return data, [i for i in result[-1] if len(i) >= threshold]

    
