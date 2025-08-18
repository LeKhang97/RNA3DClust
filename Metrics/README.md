### The Metrics used to evaluate RNA3DClust result

We've used 4 metrics to evaluate the performance of RNA3DClust, which are the **NDO, DBD and CSD**. 

**The NDO (Normalized Domain Overlap)** measures the similarity between 2 partitioning results based on the percentage of nucleotide overlapped:
![DBD_NDO_DCS(2)](https://github.com/user-attachments/assets/feb5ff20-a8a3-421c-9139-ce77fdb01d56)

Its calculation includes 2 steps: 1. Calculating the DOM (Domain Overlap Matrix); then 2. Calculating the NDO from it:
```
overlap_mtx, min_labels = domain_overlap_matrix([truth,pred],res) 
ndo_score = NDO(overlap_mtx,length_seq, min_labels)
```  
**The DBD (Domain Boundary Distance)** and **CSD (Chain Segment Distance)** measures the distances between boundaries:
![DBD_NDO_DCS(6)](https://github.com/user-attachments/assets/6f9d9883-4974-4dcb-89a3-c1ab195c7b38)

![DBD_NDO_DCS(4)](https://github.com/user-attachments/assets/c2939e51-9ff0-4f31-b405-eea48ed23f02)

Both metrics require computing the Domain Distance Matrix first. This matrix is different from each other. Then, based on the distance matrices, we calculate the final scores:

```
# Calculate the DBD score
distance_mtx = domain_distance_matrix([truth,pred],res)
dbd_score = DBD(distance_mtx, threshold=10)

# Calculate the CSD score
distance_mtx = domain_distance_matrix2([truth,pred],res)
sdd_score = CSD(distance_mtx, threshold=10)
```



