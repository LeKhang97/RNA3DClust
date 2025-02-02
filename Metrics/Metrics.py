import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib import cm

# Function to flatten a nested list
def flatten(l):
    result = []
    for sublist in l:
        if isinstance(sublist, list):
            result += sublist
        else:
            result += [sublist]
    return result

# Function to merge elements in a matrix based on given label ranges
def merge_ele_matrix(mtx, list_range_true, list_range_pred):
    merged_mtx_row = []
    for row in mtx:
        new_row = row.copy()
        for label_pos in range(len(list_range_true)):
            s = label_pos
            e = s + len(list_range_true[label_pos])
            new_row = new_row[:s] + [sum(new_row[s:e])] + new_row[e:]
        merged_mtx_row.append(new_row)
    
    merged_mtx_row = np.array(merged_mtx_row).T.tolist()
    merged_mtx = []
    for row in merged_mtx_row:
        new_row = row.copy()
        for label_pos in range(len(list_range_pred)):
            s = label_pos
            e = s + len(list_range_pred[label_pos])
            new_row = new_row[:s] + [sum(new_row[s:e])] + new_row[e:]
        merged_mtx.append(new_row)
    
    return np.array(merged_mtx).T

# Function to convert a sorted list to a list of ranges
def list_to_range(l):
    l = sorted(set(l))
    if not l:
        return []
    
    ranges = []
    s = l[0]
    for p, v in enumerate(l):
        if p >= 1 and v != l[p-1] + 1:
            ranges.append(range(s, l[p-1] + 1))
            s = v
        if p == len(l) - 1:
            ranges.append(range(s, v + 1))
    return ranges

# Function to compute domain overlap matrix between ground truth and prediction
def domain_overlap_matrix(lists_label, list_residue=None): #Order in lists_label: ground_truth, prediction
    if list_residue is None:
        list_residue = range(len(lists_label[0]))
    
    group_label = {'pred': lists_label[1], 'true': lists_label[0]}
    group_residue = {key: {label: [list_residue[i] for i in range(len(list_residue)) if group_label[key][i] == label] for label in set(group_label[key])} for key in group_label}
    
    domain_matrix = []
    for label in sorted(set(group_label['pred'])):
        row = [len([x for x in group_residue['pred'][label] if x in group_residue['true'][label2]]) for label2 in sorted(set(group_label['true']))]
        row += [0] * (len(set(group_label['pred'])) - len(set(group_label['true'])))
        domain_matrix.append(row)
    
    for _ in range(len(set(group_label['true'])) - len(set(group_label['pred']))):
        domain_matrix.append([0] * len(set(group_label['true'])))
    
    min_labels = [min(set(group_label['true'])), min(set(group_label['pred']))] # Get the minimum label for each group (if there is -1, it means there is outlier)
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

# Function to compute domain distance matrix
def domain_distance_matrix(lists_label, list_residue=None): #Order in lists_label: ground_truth, prediction
    if list_residue is None:
        list_residue = range(len(lists_label[0]))
    
    group_label = {'pred': lists_label[1], 'true': lists_label[0]}
    group_residue = {key: {label: [list_residue[i] for i in range(len(list_residue)) if group_label[key][i] == label] for label in set(group_label[key])} for key in group_label}
    
    domain_distance_mtx = []
    for label1 in sorted(set(group_label['pred'])):
        for segment1 in list_to_range(group_residue['pred'][label1]):
            row = [domain_distance(segment2, segment1) for label2 in sorted(set(group_label['true'])) for segment2 in list_to_range(group_residue['true'][label2])]
            domain_distance_mtx.append(row)
    
    lst_of_range_true = [list_to_range(group_residue['true'][label]) for label in sorted(set(group_label['true']))]
    lst_of_range_pred = [list_to_range(group_residue['pred'][label]) for label in sorted(set(group_label['pred']))]
    
    return domain_distance_mtx, lst_of_range_true, lst_of_range_pred

# Function to compute Domain Boundary Distance (DBD)
def DBD(domain_distance_mtx, list_range_true, list_range_pred, threshold=7):
    scoring_mtx = [[threshold - i if i < threshold else 0 for i in row] for row in domain_distance_mtx]
    merged_scoring_mtx = np.array(merge_ele_matrix(scoring_mtx, list_range_true, list_range_pred))
    
    max_score = np.amax(merged_scoring_mtx, axis=1) if merged_scoring_mtx.shape[0] >= merged_scoring_mtx.shape[1] else np.amax(merged_scoring_mtx, axis=0)
    return sum(max_score) / (threshold * max(merged_scoring_mtx.shape))

# Function to compute Domain Clustering Similarity (DCS)
def DCS(truth, pred, outliner=False):
    if not outliner:
        truth, pred = [i for i in truth if i != -1], [i for i in pred if i != -1]
    return (max(len(set(truth)), len(set(pred))) - abs(len(set(truth)) - len(set(pred)))) / max(len(set(truth)), len(set(pred)))
