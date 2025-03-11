![Version](https://img.shields.io/badge/Version-1.0.0-blue)
[![Platform](https://img.shields.io/badge/Platform-Linux-blueviolet)](https://evryrna.ibisc.univ-evry.fr/evryrna/RNA3DClust/home)
[![Docker supported](https://img.shields.io/badge/Docker-Supported-brightgreen)](https://github.com/LeKhang97/RNA3DClust/blob/main/Dockerfile)
[![GitHub repo](https://img.shields.io/badge/Repo-GitHub-white.svg)](https://github.com/LeKhang97/RNA3DClust/tree/main)
[![License](https://img.shields.io/badge/License-MIT-informational.svg)](https://github.com/LeKhang97/RNA3DClust/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2025.01.12.632579-yellow)](https://doi.org/10.1101/2025.01.12.632579)

# RNA3DClust: unsupervised segmentation of RNA 3D structures using density-based clustering
RNA3DClust uses a clustering-based approach to segment an input RNA 3D structure into independent regions, reminiscent of structural domains in proteins. Among the various options described below, RNA3DClust offers the possibility to generate PDB files, one for each domain found. It can also provide [PyMOL >= 2.5](https://www.pymol.org/) commands for coloring the delimited domains.

![Preview](https://evryrna2.ibisc.univ-evry.fr/RNA3DClust.png)

### Installation
There are two ways to install RNA3DClust:

1. Using Docker:  
```docker pull lequockhang/rna3dclust ```  
Then use this command below to run the docker image:  
```docker run -v `pwd`:/workdir/data lequockhang/rna3dclust [options] ```  
where `` `pwd` `` is your full path to your working directory containing input file(s).  
Here is the full command for example:  
```docker run -v `pwd`:/workdir/data lequockhang/rna3dclust -i data/4y1n.pdb -a A -v -o data/Output```

3. Using source code:  
```git clone https://github.com/LeKhang97/RNA3DClust```  
From here, you can either build it globally or in a virtual environment:

    * Build globally:  
    ```pip3 install -r requirements.txt```

    * Build in a virtual environment:  
    ```make```  
    Then you can execute the program in virtual environment by:  
    ```./venv/bin/python3 RNA3Dclust.py [options]```

### Usage
You can execute the program by:<br/>
```python3 RNA3Dclust.py -i Input -v -a M -o path_for_output ```

Type ```python3 RNA3Dclust.py -h``` for more information of the usage:
```
positional arguments:
  {D,M,A,S,C}
    D                   Arguments for DBSCAN clustering algorithm
    M                   Arguments for MeanShift clustering algorithm
    A                   Arguments for Agglomerative clustering algorithm
    S                   Arguments for Spectral clustering algorithm
    C                   Arguments for Contact map-based clustering algorithm

options:
  -h, --help            show this help message and exit
  -v, --verbose         verbose mode.

  -r, --reference       name of the reference partition file in JSON format for DCS calculation.

  -i INPUT, --input INPUT
                        input file. Must be in PDB format.

  -at, --atom_type      atom types to be considered in the analysis. The default is C3'.

  -t THRESHOLD, --threshold THRESHOLD
                        lower threshold for sequence length

  -o OUTPATH, --outpath OUTPATH
                        output path for JSON and PDB files. If not specified, the output will be saved in the current directory.

  -p PDB, --pdb PDB
                        output filename in PDB format.

  -j JSON, --json JSON
                        output filename in JSON format.

  -a {D,M,A,S,C}, --algorithm {D,M,A,S,C}
                        Clustering algorithm. Either: D (DBSCAN); M (MeanShift, default); A (Agglomerative); S (Spectral); C (Contact map-based))
```

- Each algorithm has its default parameters. For example, if you want to check the MeanShift, type ```python3 ./RNA3DClust.py M -h ``` for details. You can also change the parameters, in this case is the bandwidth (-b), by following: <br>
```python3 RNA3Dclust.py -i infile -v -a M -o . M -b 5```

### Example
Here is an example of using RNA3DClust and its output, using MeanShift clustering algorithm with default parameters (flat kernel, bandwidth = 0.2):
```
python3 RNA3Dclust.py -a M -i data/4y1n.pdb
==================
Input information:
==================
Using atom type:  C3'
Using algorithm:  MeanShift
Mode selected for MeanShift algorithm: bandwidth: 0.2, kernel type: flat
----------------------------------------
Executing MeanShift on chain A...

MeanShift algorithm converged after 24 iterations.

===================
Output information:
===================
Chain A has 3 clusters and without outliers.
Number of residues of cluster 1: 100
Cluster 1 positions:
1-70, 114-126, 234-250, 

Number of residues of cluster 2: 107
Cluster 2 positions:
127-233, 

Number of residues of cluster 3: 63
Cluster 3 positions:
71-113, 251-270, 



----------------------------------------
Executing MeanShift on chain B...

MeanShift algorithm converged after 30 iterations.

===================
Output information:
===================
Chain B has 2 clusters and without outliers.
Number of residues of cluster 1: 131
Cluster 1 positions:
1-93, 233-270, 

Number of residues of cluster 2: 139
Cluster 2 positions:
94-232, 

```
### Notes
- All parameters specified for each algorithm can only be accessed after typing its abbreviation (besides the option `-a [algorithm]`);
- The input must be in PDB format;
- There are 2 output files if the output option is chosen.
   - One file is the **JSON file**, which contains the coordinates, the residue numbers and the labels of clusters.
   - The other file contains the **command line for PyMOL GUI** to generate the clusters, which has the same name as the JSON file with the suffix '_pymolcmd.pml' (name ```outfile_pymolcmd.pml``` for example).  
     You can either run it from terminal by this command:  
`pymol outfile_pymolcmd.pml`  
if you already created alias for PyMOL.  
Alternatively, from PyMOL command line, you can try:  
```@outfile_pymolcmd ```
