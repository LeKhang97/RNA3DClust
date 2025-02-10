### The Metrics used to evaluate RNA3DClust result

We've used 3 metrics to evaluate the performance of RNA3DClust, which are the NDO, DBD, SDD and SDC. 

The NDO (Normalized Domain Overlap) measures the similarity between 2 partitioning results based on the percentage of nucleotide overlapped:
![DBD_NDO_DCS(2)](https://github.com/user-attachments/assets/feb5ff20-a8a3-421c-9139-ce77fdb01d56)

```  overlap_mtx, min_labels = domain_overlap_matrix([truth,pred],res) 
    ndo_score = NDO(overlap_mtx,length_seq, min_labels)
```  



