# RNA_Domain
This project aims to build a program to detect domains in RNA 3D structure derived from the PDB database, using several conventional clustering algorithms.

### Usage
You can execute the program by:<br>
```python3 ./Clustering.py -i infile -v -a M -o outfile  ```

Type ```./Clustering.py -h``` for more information of the usage:
```
positional arguments:
  {D,M,A,S}
    D                   Arguments for DBSCAN algorithm
    M                   Arguments for MeanShift algorithm
    A                   Arguments for Agglomerative Clustering algorithm
    S                   Arguments for Spectral Clustering algorithm

options:
  -h, --help            show this help message and exit
  -v, --verbose         verbose mode.
  -i INPUT, --input INPUT
                        input file. Must be in pdb format.
  -c, --chain           process all chains at once. If not, the program will process each chain individually.
  -t THRESHOLD, --threshold THRESHOLD
                        Lower threshold for sequence length
  -o OUTFILE, --outfile OUTFILE
                        output file.
  -a {D,M,A,S}, --algorithm {D,M,A,S}
                        Clustering algorithm. Either: D (DBSCAN); M (MeanShift, default); A (Agglomerative); S (Spectral))
```
### Notes
- The input must be in pdb format;
- There are 2 output files if the output option is chosen. One file is the JSON file, which contains the coordinate, the residue number and the label of clusters. The other file contains the command line for PyMOL GUI to generate the clusters. After putting it (name ```outfile_pymolcmd``` for example) and the pdb file in the same working directory, from PyMOL GUI, you can try:
  ```@outfile_pymolcmd ```
