import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import itertools
from matplotlib import cm
import sklearn
from sklearn.cluster import DBSCAN

from Mean_shift import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from numpyencoder import NumpyEncoder
import statistics
import os
import json
import inspect

class CustomNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # Check for NumPy data types
        if isinstance(obj, (np.ndarray, np.generic)):  
            return obj.tolist()  # Convert NumPy arrays and scalars to lists
            
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)  # Convert NumPy floats to Python floats
            
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)  # Convert NumPy ints to Python ints
            
        # Add any other NumPy types you want to handle here
        return super(CustomNumpyEncoder, self).default(obj)
    
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

def process_pdb(list_format, atom_type = 'C3', models = True, get_res = False):
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

        if "ATOM" in line[:6].replace(" ","") and (len(line[17:20].replace(" ","")) == 1 or line[17:20].replace(" ","")[0] == "D") and atom_type in line[12:16]:
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
    l = sorted(set(l))  # Sort and remove duplicates, but keep order
    l2 = []
    s = l[0]  # Start of the first range

    for p, v in enumerate(l):
        if p >= 1:
            # If current number is consecutive with the previous one
            if v == l[p-1] + 1:
                # If it's the last element, append the final range
                if p == len(l) - 1:
                    l2.append(range(s, v+1))
                continue
            
            # If the sequence breaks, append the current range
            e = l[p-1] + 1
            l2.append(range(s, e))
            s = v  # Start a new range with the current element

        # If it's the last element and not part of a consecutive sequence
        if p == len(l) - 1:
            l2.append(range(s, v+1))
    
    return l2


'''def pymol_process(pred, res_num, name=None, color=None):
    if color is None:
                color = ['red', 'green', 'yellow', 'orange', 'blue', 'pink', 'cyan', 'purple', 'white', 'grey', 
                         'brown','lightblue', 'lightorange', 'lightpink', 'gold']

    label_set = list(set(pred))
    
    # Repeat the color list if the number of clusters exceeds the number of colors
    color = list(itertools.islice(itertools.cycle(color), len(label_set)))

    cmd = []
    for num, label in enumerate(label_set):
        label1 = [res_num[p] for p, v in enumerate(pred) if v == label]
        clust_name = name + f'_cluster_{num}' if name is not None else f'cluster_{num}'
        cmd.append(command_pymol(label1, clust_name, color[num]))

    return cmd'''

def generate_colors(num_colors):
    colormap = cm.get_cmap('hsv', num_colors)
    return [colormap(i) for i in range(num_colors)]

def pymol_process(pred, res_num, name=None, color=None):
    print(len(pred), len(res_num))
    if color is None:
        color = ['red', 'green', 'yellow', 'orange', 'blue', 'pink', 'cyan', 'purple', 'white', 'grey', 
                    'brown','lightblue', 'lightorange', 'lightpink', 'gold']

    label_set = list(set(pred))

    if len(label_set) > len(color):
        # Generate additional colors dynamically
        additional_colors = generate_colors(len(label_set) - len(color))
        # Convert additional colors from RGBA to hex format
        additional_colors_names = ['{:02d}'.format(i) for i in range(len(color), len(label_set))]
        color.extend(additional_colors_names)

    cmd = []
    for num, label in enumerate(label_set):
        label1 = [res_num[p] for p, v in enumerate(pred) if v == label]
        clust_name = name + f'_cluster_{num}' if name is not None else f'cluster_{num}'
        if label == -1:
            cmd.append(command_pymol(label1, clust_name,'grey'))
        else:
            cmd.append(command_pymol(label1, clust_name, color[num]))

    return cmd

def command_pymol(l, name, color):
    l2 = list_to_range(l)
    mess = f'select {name}, res '
    for p,r in enumerate(l2):
        if len(r) > 1:
            mess += f'{r[0]}-{r[-1]}'
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
        if args[2] > 1:
            model = MeanShift(bandwidth = args[2], kernel = args[3], adaptive_bandwidth = args[4], cluster_all = False)
        else:
            model = MeanShift(quantile = args[2], kernel = args[3], adaptive_bandwidth = args[4], cluster_all=False)
    elif args[1] == 'A':
        print("Executing Agglomerative...")
        model = AgglomerativeClustering(n_clusters= args[2], distance_threshold= args[3])
    elif args[1] == 'S':
        print("Executing Spectral...")
        model = SpectralClustering(n_clusters= args[2], gamma= args[3])

    elif args[1] == 'C':
        print("Executing Contact-based clustering...")
        pred = contact_map_algo(data)
        return pred
        
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
            group_residue[key][label] = [list_residue[i] for i in range(len(list_residue)) if group_label[key][i] == label]

    domain_matrix = []
    # Compare if pred has more cluster than ref:
    comp_pred_ref = len(set(group_label['pred'])) - len(set(group_label['true']))
    for label in sorted(set(group_label['pred'])): 
        row = []
        for label2 in sorted(set(group_label['true'])):
            row += [len([intersect for intersect in group_residue['pred'][label] if intersect in group_residue['true'][label2]])]
        if comp_pred_ref > 0:
            row += [0]*(comp_pred_ref)
        domain_matrix += [row]
    
    if comp_pred_ref < 0:
        for i in range(-comp_pred_ref):
            domain_matrix += [[0]*len(set(group_label['true']))]
    
    min_labels = [min(set(group_label['pred'])), min(set(group_label['true']))]
    #return domain_matrix
    return domain_matrix, min_labels

def NDO(domain_matrix, len_rnas, min_labels = [0,0]):
    #print(domain_matrix, [min_labels[0]**2, min_labels[1]**2])
    domain_matrix_no_linker = [row[(min_labels[1] == -1):] for row in domain_matrix[(min_labels[0] == -1):]]
    domain_matrix_no_linker = np.asarray(domain_matrix_no_linker)
    domain_matrix = np.asarray(domain_matrix)
    
    sum_col = np.sum(domain_matrix, axis = 0)
    
    sum_row =  np.sum(domain_matrix, axis = 1)
    
    max_col = np.amax(domain_matrix_no_linker, axis = 0)
    
    max_row = np.amax(domain_matrix_no_linker, axis = 1)
    
    Y = 0
    #print(domain_matrix,domain_matrix_no_linker, sum_col, sum_row, max_col, max_row, sep='\n')
    for row in range(domain_matrix_no_linker.shape[0]):

        Y += 2*max_row[row] - sum_row[row+(min_labels[0] == -1)]
        
    for col in range(domain_matrix_no_linker.shape[1]):
        Y += 2*max_col[col] - sum_col[col+(min_labels[1] == -1)]

    score = Y/(2*(len_rnas - sum_col[0]*(min_labels[1] == -1)))
    
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


'''def DBD(lists_label, list_residue = None, threshold = 8):
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
    return score'''

def post_process(cluster_list, res_list = False):
    cluster_list2 = cluster_list.copy()
    if res_list == False:
        res_list = list(range(len(cluster_list2)))
    
    for t in range(6):
        #ranges contain clusters with ranges, values contain clusters with values
        min_cluster = (min(set(cluster_list2)))**2
        list_of_ranges = [list_to_range([v for p,v in enumerate(res_list) if cluster_list2[p] == label]) for label in sorted(set(cluster_list2))]
        list_of_values = [[v for p,v in enumerate(res_list) if cluster_list2[p] == label] for label in sorted(set(cluster_list2))]
        
        pos_ranges = 0
        
        pos_to_change = []
        for ranges in list_of_ranges:
            for subranges in ranges:
                c1 = -1; c2 = 9999; label1 = False; label2 = False
                for values in list_of_values:
                    #if len(subranges) < 100:
                    if True:
                        # If the residues right before the selected segment exists, label the segment and cluster that contains it
                        for t1 in range(1,6):
                            if subranges[0] - t1 in values:
                                c1 = list_of_values.index(values) 
                                l1 = [l for l in list_of_ranges[c1] if subranges[0] - t1 in l][0]
                                label1 = True
                                break

                        # If the residues right after the selected segment exists, label the segment and cluster that contains it
                        for t2 in range(1,6):
                            if subranges[-1] + t2 in values:
                                c2 = list_of_values.index(values)
                                l2 = [l for l in list_of_ranges[c2] if subranges[-1] + t2 in l][0]
                                label2 = True
                                break

                # If the selected segment is an outliner segment
                if min_cluster == 1 and pos_ranges == 0:
                    # If 2 segments on both sides have the same label, label the outliner that label if it's < 30
                    if c1 == c2:
                        val = list(sorted(set(cluster_list2)))[c1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])

                    elif label1 and label2:
                        val1 = list(sorted(set(cluster_list2)))[c1]
                        val2 = list(sorted(set(cluster_list2)))[c2]
                        
                        if val1 != -1 and val2 != -1:
                            if len(subranges) == 1:
                                pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])

                            elif len(subranges) < 10:
                                pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in subranges[:int(len(subranges)/2)]])
                                pos_to_change += flatten([[(j, val2) for j in range(len(res_list)) if res_list[j] == i] for i in subranges[:int(len(subranges)/2)]])
                        
                        elif val1 == -1:
                            pos_to_change += flatten([[(j, val2) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])
                            
                        elif val2 == -1:
                            val = list(sorted(set(cluster_list2)))[c1]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])
                            
                    elif label1:
                        if len(subranges) not in range(10,100) and len(l1) >= 30:
                            val = list(sorted(set(cluster_list2)))[c1]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])
                    
                    elif label2:
                        if len(subranges) not in range(10,100) and len(l2) >= 30:
                            val = list(sorted(set(cluster_list2)))[c2]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])

                # If the selected segment is not an outliner segment
                else:
                    if c1 == c2:
                        val = list(sorted(set(cluster_list2)))[c1]
                        if val == -1 and len(l1) >= 30 and len(l2) >= 30:
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])

                        elif val != -1 and len(l1) + len(l2) >= len(subranges) and len(subranges) < 30:
                            val = list(sorted(set(cluster_list2)))[c1]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])


                    elif label1 and label2:
                        val1 = list(sorted(set(cluster_list2)))[c1]
                        val2 = list(sorted(set(cluster_list2)))[c2]
                        if val1 != -1 and val2 != -1:
                            if len(l1) >= 30 and len(l2) >= 30:
                                pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in subranges[:int(len(subranges)/2)]])
                                pos_to_change += flatten([[(j, val2) for j in range(len(res_list)) if res_list[j] == i] for i in subranges[:int(len(subranges)/2)]])

                        elif val1 == -1 and len(subranges) < 30 and len(l2) >= 30:
                            pos_to_change += flatten([[(j, val2) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])

                        elif val2 == -1 and len(subranges) < 30 and len(l1) >= 30:
                             pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])
                            
                    elif label1:
                        if len(l1) > len(subranges):
                            val = list(sorted(set(cluster_list2)))[c1]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])

                    elif label2:
                        if len(l2) > len(subranges):
                            val = list(sorted(set(cluster_list2)))[c2]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])

                                    
            pos_ranges += 1
        
        for pos,val in pos_to_change:
            cluster_list2[pos] = val
    
    return cluster_list2    

def merge_ele_matrix(mtx, list_range_true, list_range_pred):
    merged_mtx_row = []
    for row in mtx:
        new_row = row.copy()  # Create a copy of the row to avoid modifying the original row
        for label_pos in range(len(list_range_true)):
            s = label_pos
            e = s + len(list_range_true[label_pos])
            new_row = new_row[:s] + [sum(new_row[s:e])] + new_row[e:]
        merged_mtx_row.append(new_row)
    
    merged_mtx_row = np.array(merged_mtx_row).T.tolist()
    
    merged_mtx = []
    for row in merged_mtx_row:
        new_row = row.copy()  # Create a copy of the row to avoid modifying the original row
        for label_pos in range(len(list_range_pred)):
            s = label_pos
            e = s + len(list_range_pred[label_pos])
            new_row = new_row[:s] + [sum(new_row[s:e])] + new_row[e:]
        merged_mtx.append(new_row)

        
    merged_mtx = np.array(merged_mtx).T  # Transpose to get the correct shape
    
    return merged_mtx

def domain_distance(segment1, segment2):
    d = abs(min(segment1) - min(segment2))
    d += abs(max(segment1) - max(segment2))
    
    return d/2

def domain_distance_matrix(lists_label, list_residue = None):#Order in lists_label: ground_truth, prediction 
    if list_residue == None:
        list_residue = range(len(lists_label[0]))
    
    group_label = {'pred': lists_label[1], 'true': lists_label[0]}

    group_residue = {'pred': {}, 'true': {}}
    for key in group_label.keys():
        for label in set(group_label[key]):
            group_residue[key][label] = [list_residue[i] for i in range(len(list_residue)) if group_label[key][i] == label]
    
    domain_distance_mtx = []
    for label1 in sorted(set(group_label['pred'])):
        for segment1 in list_to_range(group_residue['pred'][label1]):
            row = []
            for label2 in sorted(set(group_label['true'])):
                for segment2 in list_to_range(group_residue['true'][label2]):
                    #print(segment1, segment2)
                    x = domain_distance(segment2, segment1)
                    row += [x]
                    #print(x, end = '\n\n')
            
            domain_distance_mtx += [row]
    
    lst_to_range_true = [list_to_range(group_residue['true'][label1]) for label1 in sorted(set(group_label['true']))]
    lst_to_range_pred = [list_to_range(group_residue['pred'][label1]) for label1 in sorted(set(group_label['pred']))]
    
    return domain_distance_mtx, lst_to_range_true, lst_to_range_pred

def DBD(domain_distance_mtx, list_range_true, list_range_pred, threshold = 50):
    scoring_mtx = []
    for row in domain_distance_mtx:
        scoring_mtx += [[threshold - i if i < threshold else 0 for i in row]]
    
    merged_scoring_mtx = merge_ele_matrix(scoring_mtx, list_range_true, list_range_pred)

    merged_scoring_mtx = np.asarray(merged_scoring_mtx)
    scoring_mtx = np.asarray(scoring_mtx)
    #print(merged_scoring_mtx, scoring_mtx, sep = '\n')
    if scoring_mtx.shape[0] >= scoring_mtx.shape[1]:
        max_row = np.amax(merged_scoring_mtx, axis = 1)
        total_score = sum(max_row)/(threshold*scoring_mtx.shape[0])
    else:
        max_col = np.amax(merged_scoring_mtx, axis = 0)
        total_score = sum(max_col)
        total_score = sum(max_col)/(threshold*scoring_mtx.shape[1])
    
    return total_score

def DCS(truth,pred, outliner = False):
    if outliner == False:
        truth = [i for i in truth if i != -1]
        pred =  [i for i in pred if i != -1]
    
    count_truth = len(set(truth))
    count_pred = len(set(pred))
    
    DCS = (max((count_truth), count_pred) - abs(count_truth - count_pred))/max(count_truth, count_pred)
    
    return DCS

def inf(tup_of_tups, val):
    start_eles = [min(tup) for tup in tup_of_tups if min(tup) <= val]
    end_eles = [max(tup) for tup in tup_of_tups if max(tup) <= val]
    
    inf_start = max(start_eles) if bool(start_eles) != False else -1
    inf_end = max(end_eles) if bool(end_eles) != False else -1

    return inf_start, inf_end

def sup(tup_of_tups, val):
    start_eles = [min(tup) for tup in tup_of_tups if min(tup) >= val]
    end_eles = [max(tup) for tup in tup_of_tups if max(tup) >= val]
    
    sup_start = min(start_eles) if bool(start_eles) != False else val
    sup_end = min(end_eles) if bool(end_eles) != False else val

    return sup_start, sup_end

def find_tuple_index(tuples, element, order = False):
    for i, t in enumerate(tuples):
        if element in t:
            if order == False:
                return i
            else:
                if order == "first":
                    if element == t[0]:
                        return i
                else:
                    if element == t[-1]:
                        return i
    return -1  # Return -1 if the element is not found in any tuple

'''def top_down_algo(tup_of_tups, pos_list):
    tup_of_tups = tuple(tup_of_tups)
    pos_list = sorted(pos_list)
    #l = [largest_smaller_than(acc_sum_list, pos) for pos in pos_list]
    new_tups = []
    old_tup_of_tups = tup_of_tups
    #print(tup_of_tups)
    for ele in pos_list:
        len_list = [tup[-1] - tup[0] for tup in tup_of_tups]
        acc_sum_list = list(itertools.accumulate(len_list))
        
        inf_start, inf_end = inf(tup_of_tups, ele)
        sup_start, sup_end = sup(tup_of_tups, ele)
        X = [i for i in range(len(acc_sum_list)) if acc_sum_list[i] < ele]
        #ind_inf_end = find_tuple_index(tup_of_tups, inf_end)
        ind_inf_end = max(X) if bool(X) else -1
        #print(acc_sum_list, ele,ind_inf_end)

        if ind_inf_end == -1:
            x = (tup_of_tups[ind_inf_end+1][0], tup_of_tups[ind_inf_end+1][0] + ele - 0)
            y = (tup_of_tups[ind_inf_end+1][0] + ele - 0, sup_end)
            tup_of_tups = tuple( tuple([x,y]) + tup_of_tups[1:])
        else:

            x = (tup_of_tups[ind_inf_end+1][0], tup_of_tups[ind_inf_end+1][0] + ele - acc_sum_list[ind_inf_end]) # need to fix
            y = (tup_of_tups[ind_inf_end+1][0] + ele - acc_sum_list[ind_inf_end], tup_of_tups[ind_inf_end+1][1]) # need to fix
            tup_of_tups = tuple(tup_of_tups[:ind_inf_end+1] +  tuple([x,y]) + tup_of_tups[ind_inf_end + 2:])
        
        #print('x,y:', x,y)
        
        #print(tup_of_tups)
        tup_of_tups = tuple([i for i in tup_of_tups if i[0] != i[-1]])
        
    new_tups = tuple(i for i in tup_of_tups if i not in old_tup_of_tups)

    print(new_tups)
    if len(new_tups) == 2:
        tups1 = tup_of_tups[:tup_of_tups.index(new_tups[0])+1]
        tups2 = tup_of_tups[tup_of_tups.index(new_tups[1]):]
    
    elif len(new_tups) == 3:
        tups1 = tup_of_tups[:tup_of_tups.index(new_tups[0])+1] + tup_of_tups[tup_of_tups.index(new_tups[2]):]
        tups2 = tup_of_tups[tup_of_tups.index(new_tups[1]):tup_of_tups.index(new_tups[2])]
        
    elif len(new_tups) == 4:
        tups1 = tup_of_tups[:tup_of_tups.index(new_tups[0])+1] + tup_of_tups[tup_of_tups.index(new_tups[3]):]
        tups2 = tup_of_tups[tup_of_tups.index(new_tups[1]):tup_of_tups.index(new_tups[2])+1]
    
    elif len(new_tups) == 1:
        tups1 = tup_of_tups[:tup_of_tups.index(new_tups[0])]
        tups2 = tup_of_tups[tup_of_tups.index(new_tups[0])+1:]
    
    else:
        tups1 = tup_of_tups[:ind_inf_end+2]
        tups2 = tup_of_tups[ind_inf_end+2:]

    return tups1,tups2'''

import itertools

def top_down_algo(tup_of_tups, pos_list):
    # Convert tuples to lists for faster updates
    tup_of_tups = list(tup_of_tups)
    pos_list = sorted(pos_list)
    
    new_tups = []
    old_tup_of_tups = tup_of_tups.copy()
    
    # Calculate the initial len_list and acc_sum_list just once
    len_list = [tup[-1] - tup[0] for tup in tup_of_tups]
    acc_sum_list = list(itertools.accumulate(len_list))
    
    for ele in pos_list:
        # Only update acc_sum_list incrementally when necessary
        X = [i for i in range(len(acc_sum_list)) if acc_sum_list[i] < ele]
        ind_inf_end = max(X) if X else -1

        if ind_inf_end == -1:
            # Handle case where ind_inf_end == -1
            x = (tup_of_tups[0][0], tup_of_tups[0][0] + ele)
            y = (tup_of_tups[0][0] + ele, tup_of_tups[0][1])
            tup_of_tups[0] = x
            tup_of_tups.insert(1, y)
        else:
            # Split tuple based on cumulative sum
            x = (tup_of_tups[ind_inf_end+1][0], tup_of_tups[ind_inf_end+1][0] + ele - acc_sum_list[ind_inf_end])
            y = (tup_of_tups[ind_inf_end+1][0] + ele - acc_sum_list[ind_inf_end], tup_of_tups[ind_inf_end+1][1])
            tup_of_tups[ind_inf_end+1] = x
            tup_of_tups.insert(ind_inf_end+2, y)

        # Remove empty tuples in place to avoid redundant filtering
        tup_of_tups = [tup for tup in tup_of_tups if tup[0] != tup[-1]]
        
        # Update len_list and acc_sum_list incrementally
        len_list = [tup[-1] - tup[0] for tup in tup_of_tups]
        acc_sum_list = list(itertools.accumulate(len_list))
    
    new_tups = [i for i in tup_of_tups if i not in old_tup_of_tups]
    
    print(new_tups)
    
    # Generate tups1 and tups2 based on the length of new_tups
    if len(new_tups) == 2:
        tups1 = tup_of_tups[:tup_of_tups.index(new_tups[0])+1]
        tups2 = tup_of_tups[tup_of_tups.index(new_tups[1]):]
    
    elif len(new_tups) == 3:
        tups1 = tup_of_tups[:tup_of_tups.index(new_tups[0])+1] + tup_of_tups[tup_of_tups.index(new_tups[2]):]
        tups2 = tup_of_tups[tup_of_tups.index(new_tups[1]):tup_of_tups.index(new_tups[2])]
        
    elif len(new_tups) == 4:
        tups1 = tup_of_tups[:tup_of_tups.index(new_tups[0])+1] + tup_of_tups[tup_of_tups.index(new_tups[3]):]
        tups2 = tup_of_tups[tup_of_tups.index(new_tups[1]):tup_of_tups.index(new_tups[2])+1]
    
    elif len(new_tups) == 1:
        tups1 = tup_of_tups[:tup_of_tups.index(new_tups[0])]
        tups2 = tup_of_tups[tup_of_tups.index(new_tups[0])+1:]
    
    else:
        tups1 = tup_of_tups[:ind_inf_end+2]
        tups2 = tup_of_tups[ind_inf_end+2:]

    return tuple(tups1), tuple(tups2)


def bot_up_algo(fragments, fragment_indexes):
    frags = fragments.copy()
    fragment_inds =  tup_pos_process(fragment_indexes.copy())
    
    while True:
        flag = 1
        new_frags = frags; new_fragment_inds = fragment_inds
        for ind1, ind2 in itertools.combinations(range(len(frags)), 2):
            S = (DISinter(frags[ind1], frags[ind2]) - min(DISintra(fragment_inds[ind1], fragment_inds[ind1]), 
                                                                 DISintra(fragment_inds[ind2], fragment_inds[ind2])))
            #print(ind1,ind2,S)
            if S >= 0:
                new_fragment_inds = [inds for inds in new_fragment_inds if inds != fragment_inds[ind1] and inds != fragment_inds[ind2]]
                new_fragment_inds += [fragment_inds[ind1] + fragment_inds[ind2]]
                
                new_frags += [np.concatenate((frags[ind1], frags[ind2]), axis=0)]
                
                new_frags = [arr for arr in new_frags if not np.array_equal(arr, frags[ind1]) and not np.array_equal(arr, frags[ind2])]

                
                frags = new_frags
                fragment_inds = new_fragment_inds
                flag = 0
                break
                
        if flag == 1:
            return fragment_inds

def contact_prob(d, d0 = 8, sig = 1.5):
    p = 1/(1+np.exp((d - d0)/sig))
    
    return p

def DISinter(D1, D2, alpha=0.43, d0=8, sig=1.5):
    D1 = np.array(D1)
    D2 = np.array(D2)

    dists = np.linalg.norm(D1[:, np.newaxis] - D2, axis=2)
    probs = contact_prob(dists, d0, sig)

    s = np.sum(probs)
    s *= 1 / ((len(D1) * len(D2)) ** alpha)

    return s

def DISintra(D, indices = None, beta = 0.95, d0 = 8, sig = 1.5):
    l = len(D)
    if indices == None:
        indices = range(len(D))
    
    s = 0
    for i1, i2 in itertools.combinations(range(len(indices)),2):
        if abs(indices[i1]-indices[i2]) <= 2:
            continue
        
        d1 = np.array(D[i1]); d2 = np.array(D[i2])
        
        d = np.linalg.norm(d2 - d1)
        p = contact_prob(d,d0,sig)
        
        s += p
    
    s *= 1/(l**beta)
    
    return s

def largest_smaller_than(lst, value):
    # Initialize variables to store the largest element found and its index
    largest = None
    largest_index = -1
    
    # Iterate through the list with index
    for index, elem in enumerate(lst):
        # Check if the element is smaller than the given value
        if elem <= value:
            # If largest is None or current element is larger than largest found so far
            if largest is None or elem > largest:
                largest = elem
                largest_index = index
    
    return largest, largest_index,value

def tup_pos_process(tup_of_tup):
    result = []
    for tup in tup_of_tup:
        s = []
        for i in tup:
            s += list(range(i[0], i[1]))
        
        s = tuple(s)
        result += [s]
    
    return result

'''def contact_map_algo(original_frags):
    frags = [original_frags.copy()]
    frag_ind= [tuple([(0,len(original_frags))])]
    s = 0
    while True:
        flag = 0
        new_frags = []
        new_frag_ind = []
        for frag,ind in zip(frags,frag_ind):
            DIS2 = {}
            for position1 in range(30,len(frag)-30):
                for position2 in range(position1+30,len(frag)-30):
                    if position2 == None:
                        continue
                    d = np.linalg.norm(np.array(frag[position1]) - np.array(frag[position2]))
                    if d < 14:
                        DIS2[position1, position2] = DISinter(np.concatenate((frag[:position1], frag[position2:]), axis=0), frag[position1:position2])
                    for position in [position1, position2]:
                        if (position,) not in DIS2.keys():
                            DIS2[position,] = DISinter(frag[:position], frag[position:])             

            if bool(DIS2.keys()) == False:
                new_frag_ind += tuple([ind])
                new_frags += [frag]
                continue

            min_key = min(DIS2, key=DIS2.get)
            min_value = DIS2[min_key].copy()

            if min_value < DISintra(frag)/2:
                print('index and key: ', ind, min_key)
                x = top_down_algo(ind, min_key)
                print(x)
                new_frag_ind += x
                flag = 1
                new_frags += [np.concatenate([original_frags[i[0]:i[1]] for i in m], axis = 0) for m in x]
                
            else:
                new_frag_ind += tuple([ind])
                new_frags += [frag]

        frag_ind = new_frag_ind
        frags = new_frags

        #print(frag_ind)
        s += 1
        if flag == 0:
            result = bot_up_algo(frags, frag_ind)
            return result'''

def contact_map_algo(original_frags):
    frags = [original_frags.copy()]
    frag_ind = [tuple([(0, len(original_frags))])]
    s = 0

    while True:
        flag = 0
        new_frags = []
        new_frag_ind = []

        for frag, ind in zip(frags, frag_ind):
            # Pre-compute distance matrix for efficiency
            distances = np.array([[np.linalg.norm(frag[i] - frag[j]) for j in range(len(frag))] for i in range(len(frag))])

            DIS2 = {}
            for position1 in range(30, len(frag) - 30):
                for position2 in range(position1 + 30, len(frag) - 30):
                    if distances[position1, position2] < 8:
                        DIS2[position1, position2] = DISinter(np.concatenate((frag[:position1], frag[position2:]), axis=0), frag[position1:position2])

            if not DIS2:
                new_frag_ind.append(ind)
                new_frags.append(frag)
                continue

            min_key = min(DIS2, key=DIS2.get)
            min_value = DIS2[min_key].copy()

            if min_value < DISintra(frag) / 2:
                print('index and key:', ind, min_key)
                x = top_down_algo(ind, min_key)
                print(x)
                new_frag_ind += x
                flag = 1
                new_frags += [np.concatenate([original_frags[i[0]:i[1]] for i in m], axis=0) for m in x]
            else:
                new_frag_ind.append(ind)
                new_frags.append(frag)

        frag_ind = new_frag_ind
        frags = new_frags

        s += 1
        if flag == 0:
            print('perform bot_up_algo')
            result = bot_up_algo(frags, frag_ind)
            return result

def process_cluster_format(clust_lst, res_lst = None):
    if res_lst == None:
        res_lst = list(range(1,len(clust_lst)+1))

    clust_by_res = []
    set_clust = set(clust_lst)
    for clust in set_clust:
        sublst = []
        for pos, res in enumerate(res_lst):
            if clust_lst[pos] ==  clust:
                sublst += [res]

        clust_by_res += [sublst]

    return clust_by_res
    
def split_pdb_by_clusters(pdb_file, clusters, output_prefix, chain=None):
    """
    Splits a PDB file into multiple PDB files based on provided clusters of residues.
    If a chain is specified, only residues from that chain will be processed.

    Parameters:
        pdb_file (str): Path to the input PDB file.
        clusters (list of list of int): List of clusters, where each cluster is a list of residue indices.
        output_prefix (str): Prefix for output files.
        chain (str, optional): Chain ID to filter residues by. If None, all chains are processed.
    """

    # Read the PDB file
    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    # Create a dictionary to store lines for each cluster
    cluster_lines = {i: [] for i in range(len(clusters))}

    # Process each line in the PDB file
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # Extract residue sequence number and chain ID
            residue_seq = int(line[22:26].strip())  # Extract residue sequence number
            residue_chain = line[21:22].strip()  # Extract chain ID (column 22)

            # If a chain is specified, skip lines that don't match the chain
            if chain and residue_chain != chain:
                continue

            # Check which cluster this residue belongs to
            for cluster_index, cluster in enumerate(clusters):
                if residue_seq in cluster:
                    cluster_lines[cluster_index].append(line)
                    break
    
    return cluster_lines