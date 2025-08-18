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
from sklearn.metrics import pairwise_distances
from Bio.PDB import PDBParser, MMCIFParser
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
    l = [(0,6),(6,11),(12,16),(16,17),(17,20),(20,22),(22,26),
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

        model = '' # Adapt the web-server
        if ("ATOM" in line[:6].replace(" ","") and (len(line[17:20].replace(" ","")) == 1 or line[17:20].replace(" ","")[0] == "D")) and atom_type in line[12:16]:
            new_line = [line[v[0]:v[1]].replace(" ","") for v in l ] + [model]

            chain_name = new_line[5]
            if chain_name not in chains:
                chains += [chain_name]

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
    
    print(set(chains))
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

def cluster_algo(*args):
    print(args[1:])
    data = args[0]
    print('-'*40)
    if args[1] == 'D':
        print(f"Executing DBSCAN on chain {args[-1]}...")
        if args[3] <= 1 and type(args[3]) == float:
            min_samples = int(args[3] * len(data))
            if min_samples == 0:
                min_samples = 1
        else:
            min_samples = args[3]
        if args[2] <= 1 and type(args[2]) == float:
            eps = pairwise_distances(data, metric='euclidean').max() * args[2]
        else:
            eps = args[2]

        print(f"Using eps = {eps}, min_samples = {min_samples}")
        model = DBSCAN(eps=eps, min_samples= min_samples)
    elif args[1] == 'M':
        print(f"Executing MeanShift on chain {args[-1]}...")
        if args[2] > 1:
            model = MeanShift(bandwidth = args[2], kernel = args[3], bandwidth_mode= args[4], cluster_all = False, max_iter= 300)
        else:
            model = MeanShift(quantile = args[2], kernel = args[3], bandwidth_mode= args[4], cluster_all = False, max_iter= 300)
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
    
    n = 11 # Number of iterations
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
                        for t1 in range(1,7):
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
                            #if len(l1) > len(subranges) and len(l2) > len(subranges) and len(subranges) < 30:
                                pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in subranges[:int(len(subranges)/2)]])
                                pos_to_change += flatten([[(j, val2) for j in range(len(res_list)) if res_list[j] == i] for i in subranges[:int(len(subranges)/2)]])

                        elif val1 == -1 and len(subranges) < 30 and len(l2) >= 30:
                        #elif val1 == -1 and len(subranges) < 30 and len(l2) > len(subranges) and len(l1) > len(subranges):
                            pos_to_change += flatten([[(j, val2) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])

                        elif val2 == -1 and len(subranges) < 30 and len(l1) >= 30:
                        #elif val2 == -1 and len(subranges) < 30 and len(l2) > len(subranges) and len(l1) > len(subranges):
                             pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])
                            
                    elif label1:
                        if len(l1) > len(subranges) and len(subranges) < 30 and len(l1) >= 30:
                            val = list(sorted(set(cluster_list2)))[c1]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])

                    elif label2:
                        if len(l2) > len(subranges) and len(subranges) < 30 and len(l2) >= 30:
                            val = list(sorted(set(cluster_list2)))[c2]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])

                if t == n-1 and len(ranges) == 1:
                    if len(ranges[0]) < 10 and c2 != 9999:
                        val = list(sorted(set(cluster_list2)))[c2]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])
                    elif len(ranges[0]) < 30:
                        val = -1
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in subranges])

            pos_ranges += 1

        for pos,val in pos_to_change:
            cluster_list2[pos] = val
    
    return cluster_list2  

def make_list_of_ranges(res_list, cluster_list):
    list_of_ranges = []; l = [res_list[0]]
    for p,v in enumerate(res_list):
        if p == 0:
            continue
        if cluster_list[p] == cluster_list[p-1]:
            if (cluster_list[p] != -1 and res_list[p] - res_list[p-1] == 1) or cluster_list[p] == -1:
                l += [v]
        else:
            list_of_ranges += [l]
            l = [v]

        if p == len(res_list) - 1:
            list_of_ranges += [l]

    return list_of_ranges

def find_missing_res(file):
    from collections import defaultdict

    missing_residues = defaultdict(set)

    # === Step 1: Parse REMARK 465 or _pdbx_unobs_or_zero_occ_residues ===
    if file.endswith('.pdb'):
        is_remark_465 = False
        with open(file, 'r') as infile:
            for line in infile:
                if line.startswith('REMARK 465   M RES C SSSEQI'):
                    is_remark_465 = True
                    continue
                if is_remark_465 and line.startswith('REMARK 465'):
                    if len(line.strip()) < 20:
                        continue
                    try:
                        res_name = line[15:18].strip()
                        chain_id = line[19:20].strip()
                        res_num = line[22:27].strip()
                        if res_name in {'A', 'C', 'G', 'U', 'I', 'T'}:
                            missing_residues[chain_id].add(int(res_num))
                    except (ValueError, IndexError):
                        continue
                elif is_remark_465 and not line.startswith('REMARK'):
                    break

    elif file.endswith('.cif'):
        in_block = False
        with open(file, 'r') as infile:
            for line in infile:
                if '_pdbx_unobs_or_zero_occ_residues.polymer_flag' in line:
                    in_block = True
                    continue
                if in_block:
                    if line.startswith('#') or line.strip() == '' or line.startswith('_'):
                        break  # end of block
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        chain_id = parts[1]
                        res_num = parts[2]
                        comp_id = parts[3]
                        if comp_id in {'A', 'C', 'G', 'U', 'I', 'T'}:
                            try:
                                missing_residues[chain_id].add(int(res_num))
                            except ValueError:
                                continue

    # === Step 2: Detect implicit gaps ===
    with open(file, 'r') as f:
        lines = f.read().splitlines()

    try:
        _, chains, res_nums_per_chain = process_structure(lines, get_res=False, filename=file)
    except Exception as e:
        print(f"Error in process_structure: {e}")
        return dict(missing_residues)

    for ch, res_list in zip(chains, res_nums_per_chain):
        sorted_res = sorted(set(res_list))
        inferred_missing = []
        for i in range(1, len(sorted_res)):
            if sorted_res[i] - sorted_res[i - 1] > 1:
                inferred_missing += list(range(sorted_res[i - 1] + 1, sorted_res[i]))
        missing_residues[ch].update(inferred_missing)

    # Convert to sorted lists
    return {ch: sorted(list(missing)) for ch, missing in missing_residues.items()}

def post_process_update(cluster_list, res_list = False, missing_res = False):
    cluster_list2 = cluster_list.copy()
    if res_list == False:
        res_list = list(range(len(cluster_list2)))
    
    if missing_res == False:
        missing_res = [i for i in range(min(res_list), max(res_list)+1) if i not in res_list]
    s = 0
    while True:
        list_of_ranges = make_list_of_ranges(res_list, cluster_list2)
        if len(list_of_ranges) == 1:
            break

        label_segments = [cluster_list2[res_list.index(p[0])] for p in list_of_ranges] # Get the labels of the segments
        
        pos_to_change = []
        for pos, ranges in enumerate(list_of_ranges):
            label = label_segments[pos]
            # If the selected segment is an outlier segment
            if label == -1:
                # If the length of outlier segment is 1
                if len(ranges) == 1:
                    val = label_segments[pos-1]
                    pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                # If the selected outlier segment is the first or last segment
                # If the selected outlier segment is the first segment
                if pos == 0:
                    if label_segments[pos+1] != -1 and len(ranges) not in range(10,100) and len(list_of_ranges[pos+1]) >= 30:
                        val = label_segments[pos+1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])  
                
                # If the selected outlier segment is the last segment
                elif pos == len(list_of_ranges) - 1:
                    if label_segments[pos-1] != -1 and len(ranges) not in range(10,100) and len(list_of_ranges[pos-1]) >= 30:
                        val = label_segments[pos-1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])

                else:
                    # If 2 segments on both sides have the same label, label the outliner that label
                    if label_segments[pos-1] == label_segments[pos+1] and label_segments[pos-1] != -1:
                        val = label_segments[pos-1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                
                    # If 2 segments on both sides have different labels, label the outliner half-half if it's < 10
                    elif label_segments[pos-1] != label_segments[pos+1] and label_segments[pos-1] != -1 and label_segments[pos+1] != -1 and len(ranges) < 10:
                        val1 = label_segments[pos-1]
                        val2 = label_segments[pos+1]
                        if len(ranges) == 1:
                            pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                        else:
                            pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in ranges[:int(len(ranges)/2)]])
                            pos_to_change += flatten([[(j, val2) for j in range(len(res_list)) if res_list[j] == i] for i in ranges[int(len(ranges)/2):]])
                
            # If the selected segment is not an outlier segment
            else:
                # If the selected segment is the first or last segment
                # If the selected segment is the first segment
                if pos == 0:
                    if label_segments[pos+1] != -1 and len(ranges) < 30 and len(list_of_ranges[pos+1]) > len(ranges): # > 30
                        val = label_segments[pos+1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                    elif label_segments[pos+1] == -1 and len(ranges) < 30:
                        if len(list_of_ranges[pos+1]) > len(ranges):
                            val = label_segments[pos+1]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                        else:
                            if pos + 2 < len(list_of_ranges):
                                val1 = label_segments[pos]
                                val2 = label_segments[pos+2]
                                pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in list_of_ranges[pos+1][:int(len(list_of_ranges[pos+1])/2)]])
                                pos_to_change += flatten([[(j, val2) for j in range(len(res_list)) if res_list[j] == i] for i in list_of_ranges[pos+1][int(len(list_of_ranges[pos+1])/2):]])
                        
                # If the selected segment is the last segment
                elif pos == len(list_of_ranges) - 1:
                    if label_segments[pos-1] != -1 and len(ranges) < 30 and len(list_of_ranges[pos-1]) > len(ranges): # > 30
                        val = label_segments[pos-1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                    elif label_segments[pos-1] == -1 and len(ranges) < 30:
                        if len(list_of_ranges[pos-1]) > len(ranges):
                            val = label_segments[pos-1]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                        else:
                            if pos >= 2:
                                val1 = label_segments[pos-2]
                                val2 = label_segments[pos]
                                pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in list_of_ranges[pos-1][:int(len(list_of_ranges[pos-1])/2)]])
                                pos_to_change += flatten([[(j, val2) for j in range(len(res_list)) if res_list[j] == i] for i in list_of_ranges[pos-1][int(len(list_of_ranges[pos-1])/2):]])
                else:
                    if label_segments[pos-1] == label_segments[pos+1] and label_segments[pos-1] == -1 and len(ranges) < 30 and len(list_of_ranges[pos-1]) >= 30 and len(list_of_ranges[pos+1]) >= 30:
                        val = -1
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                    elif label_segments[pos-1] == label_segments[pos+1] and label_segments[pos-1] != -1 and len(ranges) < 30 and len(list_of_ranges[pos-1]) + len(list_of_ranges[pos+1]) > len(ranges):
                        val = label_segments[pos-1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])

                    elif label_segments[pos-1] != label_segments[pos+1] and label_segments[pos-1] != -1 and label_segments[pos+1] != -1 and len(ranges) < 30 and len(list_of_ranges[pos-1]) > 30 and len(list_of_ranges[pos+1]) > 30:
                        val1 = label_segments[pos-1]
                        val2 = label_segments[pos+1]
                        if len(ranges) == 1:
                            pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                        else:
                            pos_to_change += flatten([[(j, val1) for j in range(len(res_list)) if res_list[j] == i] for i in ranges[:int(len(ranges)/2)]])
                            pos_to_change += flatten([[(j, val2) for j in range(len(res_list)) if res_list[j] == i] for i in ranges[int(len(ranges)/2):]])

                    elif label_segments[pos-1] != label_segments[pos+1] and label_segments[pos-1] == -1 and label_segments[pos+1] != -1 and len(ranges) < 30 and len(list_of_ranges[pos+1]) > 30:
                        val = label_segments[pos+1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                    
                    elif label_segments[pos-1] != label_segments[pos+1] and label_segments[pos-1] != -1 and label_segments[pos+1] == -1 and len(ranges) < 30 and len(list_of_ranges[pos-1]) > 30:
                        val = label_segments[pos-1]
                        pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                    
                    elif len(ranges) < 30:
                        if len(list_of_ranges[pos-1]) > len(ranges):
                            if len(list_of_ranges[pos-1]) > len(list_of_ranges[pos+1]):
                                val = label_segments[pos-1]
                            else:
                                val = label_segments[pos+1]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])
                        elif len(list_of_ranges[pos+1]) > len(ranges):
                            val = label_segments[pos+1]
                            pos_to_change += flatten([[(j, val) for j in range(len(res_list)) if res_list[j] == i] for i in ranges])                    

        for new_pos,new_val in pos_to_change:
            cluster_list2[new_pos] = new_val

        # Check if any changes were made
        if bool(pos_to_change) == False or s == 30:
            break

        s += 1

    return cluster_list2  

def SDC(truth,pred, outliner = False):
    if outliner == False:
        truth = [i for i in truth if i != -1]
        pred =  [i for i in pred if i != -1]
    
    count_truth = len(set(truth))
    count_pred = len(set(pred))
    
    sdc = (max((count_truth), count_pred) - abs(count_truth - count_pred))/max(count_truth, count_pred)
    
    return sdc

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
    is_cif = pdb_file.endswith(".cif")
    cluster_lines = {i: [] for i in range(len(clusters))}

    if not is_cif:
        # == PDB FORMAT ==
        with open(pdb_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    residue_seq = int(line[22:26].strip())
                    residue_chain = line[21:22].strip()
                except ValueError:
                    continue
                if chain and residue_chain != chain:
                    continue
                for cluster_index, cluster in enumerate(clusters):
                    if residue_seq in cluster:
                        cluster_lines[cluster_index].append(line)
                        break
    else:
        # == CIF FORMAT ==
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("model", pdb_file)

        for model in structure:
            for ch in model:
                if chain and ch.id != chain:
                    continue
                for residue in ch:
                    res_id = residue.id[1]
                    for atom in residue:
                        if atom.get_name().strip() == "C3'":
                            line = atom.get_parent().get_resname()  # Use as a label placeholder
                            atom_line = f"ATOM  {atom.serial_number:5d} {atom.name:<4} {residue.resname:>3} {ch.id:1} {res_id:4d}   {atom.coord[0]:8.3f}{atom.coord[1]:8.3f}{atom.coord[2]:8.3f}  1.00  0.00           {atom.element:>2}\n"
                            for cluster_index, cluster in enumerate(clusters):
                                if res_id in cluster:
                                    cluster_lines[cluster_index].append(atom_line)
                                    break

    return cluster_lines


def extend_missing_res(list_label, list_residue):
    max_res = max(list_residue); min_res = min(list_residue)
    #list_residue_ext = list(range(min_res, max_res + 1))
    list_residue_ext = list(range(1, max_res + 1))

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

def process_structure(input_data, atom_type='C3\'', models=True, get_res=False, filename=None):
    """
    Handles both file paths or raw content + optional filename for format detection.
    """
    if isinstance(input_data, list):
        # Require filename for format detection
        if filename is None:
            raise ValueError("Filename must be provided when input_data is a list of lines.")

        if filename.endswith('.pdb'):
            return process_pdb(input_data, atom_type=atom_type, models=models, get_res=get_res)
        elif filename.endswith('.cif'):
            # Reconstruct CIF content to parse from string
            from io import StringIO
            parser = MMCIFParser(QUIET=True)
            cif_string = '\n'.join(input_data)
            structure = parser.get_structure('model', StringIO(cif_string))
            # Extract coordinates
            coor_atoms_C = []
            chains = []
            res_num = []
            result = []
            res = []

            for model in structure:
                for chain in model:
                    chain_id = chain.id
                    chains.append(chain_id)
                    xs, ys, zs, nums, names = [], [], [], [], []
                    for residue in chain:
                        for atom in residue:
                            if atom_type in atom.get_id():
                                x, y, z = atom.get_coord()
                                xs.append(x)
                                ys.append(y)
                                zs.append(z)
                                nums.append(residue.id[1])
                                names.append(residue.resname)
                    result.append((xs, ys, zs))
                    res_num.append(nums)
                    res.append(names)

            if get_res:
                return result, chains, res_num, res
            else:
                return result, chains, res_num
        else:
            raise ValueError("Unknown file type from filename.")

    elif isinstance(input_data, str):
        return process_structure(input_data, atom_type, models, get_res)

    else:
        raise TypeError("Input must be list of lines or a file path string.")
