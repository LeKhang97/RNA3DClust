import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import itertools
from matplotlib import cm
import sklearn
from sklearn.cluster import DBSCAN
import sklearn.cluster

from src.Mean_shift import *
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
    
def decorate_message(mess, cover_by = '='):
    print(cover_by*len(mess))
    print(mess)
    print(cover_by*len(mess))
    
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

        if ("ATOM" in line[:6].replace(" ","") and (len(line[17:20].replace(" ","")) == 1 or line[17:20].replace(" ","")[0] == "D")) and atom_type in line[12:16]:
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

def generate_colors(num_colors):
    colormap = cm.get_cmap('hsv', num_colors)
    return [colormap(i) for i in range(num_colors)]

def pymol_process(pred, res_num, name=None, color=None, verbose=False):
    if color is None:
        color = ['blue', 'yellow', 'magenta', 'orange', 'green', 'pink', 'cyan', 'purple', 'red','white', 'grey', 
                    'brown','lightblue', 'lightorange', 'lightpink', 'gold']

    label_set = list(set(pred))

    if len(label_set) > len(color):
        # Generate additional colors dynamically
        additional_colors = generate_colors(len(label_set) - len(color))
        # Convert additional colors from RGBA to hex format
        additional_colors_names = ['{:02d}'.format(i) for i in range(len(color), len(label_set))]
        color.extend(additional_colors_names)

    cmd = []
    if verbose:
        msg = 'Command for PyMOL:'
        decorate_message(msg)
    for num, label in enumerate(label_set):
        label1 = [res_num[p] for p, v in enumerate(pred) if v == label]
        if label == -1:
            clust_name = name + f'_outlier' if name is not None else f'outlier'
            cmd.append(command_pymol(label1, clust_name,'grey', verbose))
        else:
            clust_name = name + f'_cluster_{num+1}' if name is not None else f'cluster_{num+1}'
            cmd.append(command_pymol(label1, clust_name, color[num], verbose))

    return cmd

def command_pymol(l, name, color, verbose = False):
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
    if verbose:
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
    print('-'*40)
    if args[1] == 'D':
        print(f"Executing DBSCAN on chain {args[-1]}...")
        model = DBSCAN(eps=args[2], min_samples= args[3])
    elif args[1] == 'M':
        print(f"Executing MeanShift on chain {args[-1]}...")
        if args[2] > 1:
            model = MeanShift(bandwidth = args[2], kernel = args[3], cluster_all = False, max_iter= 300)
        else:
            model = MeanShift(quantile = args[2], kernel = args[3], cluster_all = False, max_iter= 300)
    elif args[1] == 'A':
        print(f"Executing Agglomerative on chain {args[-1]}...")
        model = AgglomerativeClustering(n_clusters= args[2], distance_threshold= args[3])
    elif args[1] == 'S':
        print(f"Executing Spectral on chain {args[-1]}...")
        model = SpectralClustering(n_clusters= args[2], gamma= args[3])

    elif args[1] == 'C':
        print(f"Executing Contact-based clustering on chain {args[-1]}...")
        pred = contact_map_algo(data)
        return pred
        
    else:
        print(args[1])
        sys.exit("Unrecognized algorithm!")

    pred = model.fit_predict(data)

    return pred

def check_C(result, threshold):
    data = []   
    removed_chain_index = []

    if result == False or len(result) == 0:
        return False

    else:
        for t in range(len(result[0])):
            if len(result[0][t][0]) < threshold:
                removed_chain_index += [t]
                continue
        
            l = [[result[0][t][0][i], result[0][t][1][i], result[0][t][2][i]] for i in range(len(result[0][t][0]))]
            data += [np.array(l)]

        return data, [i for i in result[-1] if len(i) >= threshold], removed_chain_index

def post_process(cluster_list, res_list = False):
    cluster_list2 = cluster_list.copy()
    if res_list == False:
        res_list = list(range(len(cluster_list2)))
    
    n = 10 # Number of iterations
    for t in range(n):
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
                        for t1 in range(1,11):
                            if subranges[0] - t1 in values:
                                c1 = list_of_values.index(values) 
                                l1 = [l for l in list_of_ranges[c1] if subranges[0] - t1 in l][0]
                                label1 = True
                                break

                        # If the residues right after the selected segment exists, label the segment and cluster that contains it
                        for t2 in range(1,7):
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
                        if len(subranges) not in range(10,100):
                            if len(subranges) < 10 or len(l1) >= 30:
                                val = list(sorted(set(cluster_list2)))[c1]
                                pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])


                    elif label2:
                        if len(subranges) not in range(10,100):
                            if len(subranges) < 10 or len(l2) >= 30:
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
                        if len(l1) > len(subranges) and len(subranges) < 30 and len(l1) >= 30:
                            val = list(sorted(set(cluster_list2)))[c1]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])

                    elif label2:
                        if len(l2) > len(subranges) and len(subranges) < 30 and len(l2) >= 30:
                            val = list(sorted(set(cluster_list2)))[c2]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])
        
            pos_ranges += 1

        for pos,val in pos_to_change:
            cluster_list2[pos] = val
    
    return cluster_list2    

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
                #print('index and key:', ind, min_key)
                x = top_down_algo(ind, min_key)
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
            #print('perform bot_up_algo')
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


def extend_missing_res(list_label, list_residue):
    max_res = max(list_residue); min_res = min(list_residue)
    list_residue_ext = list(range(min_res, max_res + 1))

    # Use a dictionary for fast lookups
    residue_to_label = dict(zip(list_residue, list_label))
    
    # Initialize the output label list
    list_label_ext = []
    last_label = list_label[0]  # Default to first label

    for i in list_residue_ext:
        if i in residue_to_label:
            last_label = residue_to_label[i]  # Update last seen label
        list_label_ext.append(last_label)  # Append last known label

    return list_label_ext, list_residue_ext