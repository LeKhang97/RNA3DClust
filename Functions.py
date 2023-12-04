import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import itertools
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from numpyencoder import NumpyEncoder
import statistics
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

def process_pdb(list_format, n = 3, models = True, get_res = False):
    coor_atoms_C = []
    chains = []
    res_num = []
    result = []
    res = []
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

        res += [[new_line[4] for new_line in coor_atoms_C if new_line[5] == chain.split('_')[0] 
                 and new_line[16] == ''.join(chain.split('_')[1:])]]
    if get_res:
        return result, chains, res_num, res
    else:
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
    if color == None    :
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

def distance_2arrays(arr1, arr2):
    dist = 1
    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            dist -= 1/len(arr1)
    
    return dist

def join_clusters(list_cluster):
    prev_result = list_cluster
    result = []
    cont = True
    while cont:
        cont = False
        for cluster1, cluster2 in itertools.combinations(prev_result, 2):
            if any(i in cluster1 for i in cluster2):
                if cluster1 in result:
                    result.remove(cluster1)
                if cluster2 in result:
                    result.remove(cluster2)
                
                result += [tuple(set(cluster1 + cluster2))]
                cont = True
        
        result = list(set(result))
        prev_result = result

    return prev_result

def cluster_algo(*args):
    data = args[0]
    if args[1] == 'D':
        print("Executing DBSCAN...")
        model = DBSCAN(eps=args[2], min_samples= args[3])
    elif args[1] == 'M':
        print("Executing MeanShift...")
        model = MeanShift(bandwidth= args[2])
    elif args[1] == 'A':
        print("Executing Agglomerative...")
        model = AgglomerativeClustering(n_clusters= args[2], distance_threshold= args[3])
    elif args[1] == 'S':
        print("Executing Spectral...")
        model = SpectralClustering(n_clusters= args[2], gamma= args[3])
    elif args[1] == 'C':
        print("Executing DBSCAN...")
        model1 = DBSCAN(eps=args[2], min_samples= args[3])
        print("Executing MeanShift...")
        model2 = MeanShift(bandwidth= args[4])
        print("Executing Agglomerative...")
        model3 = AgglomerativeClustering(n_clusters= args[5], distance_threshold= args[6])
        print("Executing Spectral...")
        model4 = SpectralClustering(n_clusters= args[6], gamma= args[7])

        pred1 = model1.fit_predict(data)
        pred2 = model2.fit_predict(data)
        pred3 = model3.fit_predict(data)
        pred4 = model4.fit_predict(data)

        cluster_list = []
        for p1, p2 in itertools.combinations(range(len(pred1)), 2):
            nu1 = [pred1[p1], pred2[p1], pred3[p1], pred4[p1]]
            nu2 = [pred1[p2], pred2[p2], pred3[p2], pred4[p2]]

            if distance_2arrays(nu1, nu2) <= 0.25:
                flag = 0
                for pos, cluster in enumerate(cluster_list):
                    if p1 in cluster:
                        cluster_list[pos] += (p2,)
                        flag = 1
                        break
                    elif p2 in cluster:
                        cluster_list[pos] += (p1,)
                        flag = 1
                        break
                
                if flag == 0:
                    cluster_list += [(p1, p2)]
                    break

        cluster_list = join_clusters(cluster_list)
        
        pred = [-1 for i in range(len(data))]
        
        for cluster in cluster_list:
            for p in cluster:
                pred[p] = cluster_list.index(cluster)
        
        return np.asarray(pred)

    else:
        print(args[1])
        sys.exit("Non recognized algorithm!")

    print(args[2], args[3])

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

    
def domain_overlap_matrix(lists_label, list_residue = None): #Order in lists_label: ground_truth, prediction 
    if list_residue == None:
        list_residue = range(len(lists_label[0]))
    
    group_label = {'pred': lists_label[1], 'true': lists_label[0]}

    group_residue = {'pred': {}, 'true': {}}

    for key in group_label.keys():
        for label in set(group_label[key]):
            #group_residue[key][label] = list_to_range([lists_residue[key][i] for i in range(len(lists_residue[key])) if lists_label[key][i] == label])
            group_residue[key][label] = [list_residue[i] for i in range(len(list_residue)) if group_label[key][i] == label]

    domain_matrix = []
    for label in set(group_label['pred']): 
        row = []
        for label2 in set(group_label['true']):
            row += [len([intersect for intersect in group_residue['pred'][label] if intersect in group_residue['true'][label2]])]

        domain_matrix += [row]

    return domain_matrix

def NDO(domain_matrix, len_domains):
    domain_matrix = np.asarray(domain_matrix)
    #print(domain_matrix)
    sum_col = np.sum(domain_matrix, axis = 0)
    
    sum_row =  np.sum(domain_matrix, axis = 1)
    
    max_col = np.amax(domain_matrix, axis = 0)
    
    max_row = np.amax(domain_matrix, axis = 1)
    
    Y = 0
    for row in range(domain_matrix.shape[0]):
        Y += 2*max_row[row] - sum_row[row]
        
    for col in range(domain_matrix.shape[1]):
        Y += 2*max_col[col] - sum_col[col]

    score = Y*100/(2*len_domains)
    
    return score

def domain_boundary(lists_label, list_residue = None): #Order in lists_label: ground_truth, prediction 
        if list_residue == None:
            list_residue = range(len(lists_label[0]))
        
        group_label = {'pred': lists_label[1], 'true': lists_label[0]}

        group_residue = {'pred': {}, 'true': {}}
        
        group_boundary = {'pred': {}, 'true': {}}
        
        for key in group_label.keys():
            list_boundary = []
            for label in set(group_label[key]):
                #group_residue[key][label] = list_to_range([lists_residue[key][i] for i in range(len(lists_residue[key])) if lists_label[key][i] == label])
                group_residue[key][label] = [ list_to_range(list_residue[i] for i in range(len(list_residue)) if group_label[key][i] == label)]
            
            
                list_boundary += flatten([[(j[0],j[-1]) for j in i] for i in group_residue[key][label]])
            
            group_boundary[key] = list_boundary
            
        return group_boundary


def DBD(lists_label, list_residue = None, threshold = 8):
    group_boundary = domain_boundary(lists_label, list_residue)
    
    num_domains = max(len(group_boundary[key]) for key in group_boundary.keys()) - 1
    
    dict_boundary = {}
    for key in group_boundary.keys():
        dict_boundary[key] = [ele[0] for ele in group_boundary[key]]
    
    score = 0
    #for i,j in itertools.product(dict_boundary['pred'], dict_boundary['true']):
    if len(dict_boundary['pred']) >= len(dict_boundary['true']):
        key1 = 'pred'; key2 = 'true'
    else:
        key2 = 'pred'; key1 = 'true'
    
    print(dict_boundary)
    for i in dict_boundary[key1]:
        max_bound_dist = 0
        for j in dict_boundary[key2]:
            if threshold - abs(i - j) >= max_bound_dist:
                max_bound_dist = threshold - abs(i - j)
                t = j
            
        if i != 0 and t != 0 and max_bound_dist != 100000:
            score += max_bound_dist
            print(max_bound_dist, i, t)
    
    score = score/(threshold*num_domains)
    
    print(num_domains, group_boundary)
    return score
