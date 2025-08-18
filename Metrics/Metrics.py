import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import itertools
from matplotlib import cm

def flatten(l):
    result = []
    for sublist in l:
        if isinstance(sublist, list):
            result += sublist
        else:
            result += [sublist]

    return result

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

def list_to_range(l):
    l = sorted(set(l))  # Sort and remove duplicates, but keep order
    l2 = []
    if len(l) == 0:
        return []
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

# Function to compute matrix for Domain Overlap (NDO)
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
    for label in sorted(set(group_label['pred'])): 
        row = []
        for label2 in sorted(set(group_label['true'])):
            row += [len([intersect for intersect in group_residue['pred'][label] if intersect in group_residue['true'][label2]])]

        domain_matrix += [row]


    min_labels = [min(set(group_label['pred'])), min(set(group_label['true']))]
    #return domain_matrix
    return domain_matrix, min_labels

# Function to calculate Normalized Domain Overlap (NDO)
def NDO(domain_matrix, len_rnas, min_labels=[0, 0]):
    domain_matrix_no_linker = np.array([row[(min_labels[1] == -1):] for row in domain_matrix[(min_labels[0] == -1):]])
    domain_matrix = np.array(domain_matrix)
    
    sum_col = np.sum(domain_matrix, axis=0)
    sum_row = np.sum(domain_matrix, axis=1)
    max_col = np.amax(domain_matrix_no_linker, axis=0)
    max_row = np.amax(domain_matrix_no_linker, axis=1)
    
    Y = sum(2 * max_row - sum_row[(min_labels[0] == -1):]) + sum(2 * max_col - sum_col[(min_labels[1] == -1):])
    score = Y / (2 * (len_rnas - sum_col[0] * (min_labels[1] == -1)))
    return score

# Function to compute domain distance
def domain_distance(segment1, segment2):
    return (abs(min(segment1) - min(segment2)) + abs(max(segment1) - max(segment2))) / 2

# Matrix for calculating Chain Segment Distance (CSD)
def domain_distance_matrix2(lists_label, list_residue=None): #Order in lists_label: ground_truth, prediction
    if list_residue is None:
        list_residue = range(len(lists_label[0]))
    
    group_label = {'pred': lists_label[1], 'true': lists_label[0]}
    outlier_pos = {'pred': list_to_range([list_residue[i] for i in range(len(list_residue)) if group_label['pred'][i] == -1]), 
                   'true': list_to_range([list_residue[i] for i in range(len(list_residue)) if group_label['true'][i] == -1])}
    
    for key in outlier_pos:
        if not bool(outlier_pos[key]):
            outlier_pos[key] = [[-9999]]
    
    group_residue = {key: {label: [list_residue[i] for i in range(len(list_residue)) if group_label[key][i] == label] for label in set(group_label[key])} for key in group_label}
    
    domain_distance_mtx = []
    for label1 in sorted(set(group_label['pred']) - {-1}):
        for segment1 in list_to_range(group_residue['pred'][label1]):
            row = []
            for label2 in sorted(set(group_label['true']) - {-1}):
                for segment2 in list_to_range(group_residue['true'][label2]):
                    score = domain_distance(segment2, segment1)
                    # Check if segment boundaries are adjacent to a linker
                    lst1_min = [min(segment1)]; lst2_min = [min(segment2)]
                    lst1_max = [max(segment1)]; lst2_max = [max(segment2)]
                    for i, j in itertools.product(outlier_pos['pred'], outlier_pos['true']):
                        # Adjust start position if next to a linkers
                        if min(segment1) == i[-1] + 1:
                            lst1_min += [i[0]]
                        if min(segment2) == j[-1] + 1:
                            lst2_min += [j[0]]
                        # Adjust end position if next to a linker
                        if max(segment1) == i[0] - 1:
                            lst1_max += [i[-1]]
                        if max(segment2) == j[0] - 1:
                            lst2_max += [j[-1]]

                    for min_pos in itertools.product(lst1_min, lst2_min):
                        for max_pos in itertools.product(lst1_max, lst2_max):
                            new_score = domain_distance(range(min_pos[1], max_pos[1] + 1),
                                                        range(min_pos[0], max_pos[0] + 1))
                        if new_score < score:
                            score = new_score

                    row.append(score)

            domain_distance_mtx.append(row)
    
    lst_of_range_true = [list_to_range(group_residue['true'][label]) for label in sorted(set(group_label['true'])) if label != -1]
    lst_of_range_pred = [list_to_range(group_residue['pred'][label]) for label in sorted(set(group_label['pred'])) if label != -1]

    return domain_distance_mtx

def remove_row_col(matrix, n, m):
    # Remove rows with indices in list n
    matrix = [row for i, row in enumerate(matrix) if i not in n]
    
    # Remove columns with indices in list m from each remaining row
    matrix = [[elem for j, elem in enumerate(row) if j not in m] for row in matrix]
    
    return matrix

# Function to compute matrix for Domain Boundary Distance (DBD)
def domain_distance_matrix(lists_label, list_residue=None): # Order in lists_label: ground_truth, prediction
    if list_residue is None:
        list_residue = range(len(lists_label[0]))

    dict_label = {'pred': lists_label[1], 'true': lists_label[0]}
    dict_label_linker_pred = dict_label.copy()

    # Convert all outlier segment in pred into domain label
    dict_label['pred'] = [i if i != -1 else max(set(dict_label['pred'])) + 1 for i in dict_label['pred']]

    dict_boundary = {}; dict_boundary_linker_pred = {}
    for key in dict_label.keys():
        dict_boundary[key] = []; dict_boundary_linker_pred[key] = []
        prev_label = dict_label[key][0]
        for pos in range(len(dict_label[key])):
            if dict_label[key][pos] != prev_label or dict_label[key][pos] == -1:
                dict_boundary[key] += [list_residue[pos]]
            if dict_label[key][pos] != prev_label or dict_label_linker_pred[key][pos] == -1:
                dict_boundary_linker_pred[key] += [list_residue[pos]]
            
            prev_label = dict_label[key][pos]
        
        dict_boundary[key] = list_to_range(dict_boundary[key])
        dict_boundary_linker_pred[key] = list_to_range(dict_boundary_linker_pred[key])
    
    # Some exceptional cases
    # If the linker is only in the first or last position of pred, and/or there is no linker or also only in the first or last position of true, return distance 0
    #if len(dict_boundary_linker_pred['pred']) <= 2 and len(dict_boundary_linker_pred['true']) <= 2:
    bool_pred = all(any(except_pos in boundary for except_pos in [list_residue[0], list_residue[-1]]) for boundary in dict_boundary_linker_pred['pred']) or len(dict_boundary_linker_pred['pred']) == 0
    bool_true = all(any(except_pos in boundary for except_pos in [list_residue[0], list_residue[-1]]) for boundary in dict_boundary_linker_pred['true']) or len(dict_boundary_linker_pred['true']) == 0
        
    if bool_pred and bool_true:
        return [[0]]
    
    elif bool_true:
        return [[999999]]
    
    elif len(dict_boundary['pred']) == 0 or len(dict_boundary['true']) == 0:
        return [[999999]]

    domain_distance_mtx = []
    for boundary1 in dict_boundary['pred']:
        row = []
        for boundary2 in dict_boundary['true']:
            min_distance = 999999
            for b1, b2 in itertools.product(boundary1, boundary2):
                distance = abs(b1 - b2)
                if distance < min_distance:
                    min_distance = distance
                if min_distance == 0:
                    break   
            row.append(min_distance)
        domain_distance_mtx.append(row)

    return domain_distance_mtx

# Function to compute Domain Boundary Distance (DBD) or Structural Domain Distance (SDD)
def SDD(domain_distance_mtx, threshold=20):
    scoring_mtx = [[threshold - i if i < threshold else 0 for i in row] for row in domain_distance_mtx]
    
    # Max values by column
    max_by_col = np.max(scoring_mtx, axis=0).tolist() 

    # Max values by row
    max_by_row = np.max(scoring_mtx, axis=1).tolist() 
    
    if len(max_by_row) >= len(max_by_col):
        dbd = sum(max_by_row)/(threshold*len(max_by_row))
    else:
        dbd = sum(max_by_col)/(threshold*len(max_by_col))

    return dbd

# Function to compute Domain Boundary Distance (DBD) or Structural Domain Distance (SDD)
def DBD(domain_distance_mtx, threshold=20):
    scoring_mtx = [[threshold - i if i < threshold else 0 for i in row] for row in domain_distance_mtx]
    
    # Max values by column
    sum_all = sum(np.sum(scoring_mtx, axis=0).tolist()) 
    max_by_col = np.max(scoring_mtx, axis=0).tolist()

    # Max values by row
    max_by_row = np.max(scoring_mtx, axis=1).tolist() 
    
    if len(max_by_row) >= len(max_by_col):
        dbd = sum_all/(threshold*len(max_by_row))
    else:
        dbd = sum_all/(threshold*len(max_by_col))

    return dbd

# Function to compute Structural Domain Count (SDC)
def SDC(truth, pred, outliner=False):
    if not outliner:
        truth, pred = [i for i in truth if i != -1], [i for i in pred if i != -1]
    return (max(len(set(truth)), len(set(pred))) - abs(len(set(truth)) - len(set(pred)))) / max(len(set(truth)), len(set(pred)))

