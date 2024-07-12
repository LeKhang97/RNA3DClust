# RNA_Domain
This project aims to build a program to detect domains in RNA 3D structure derived from the PDB database, using several conventional clustering algorithms.

### Prerequisites
PyMOL version 2.5 or later

### Installation
There are 2 ways to install the tool:

#### 1.  Using Docker:
```docker pull lequockhang/rna3dclust ```

Then use this command below to run the docker image:
``` docker run -v `pwd`:/workdir/ lequockhang/rna3dclust [options] ```

Whereas `` `pwd` `` is your full path to your working directory containing input file(s). Here is the full command for example:

``` docker run -v `pwd`:/workdir/ lequockhang/rna3dclust -i Example.pdb -a A -v -o Output```

#### 2.  Using source code:
```git clone https://github.com/LeKhang97/RNA3DClust```

From here, you can either build it globally or in a virtual environment:

##### 2.1 Build globally:
```pip3 install -r requirements.txt```

##### 2.2 Build in a virtual environment:
```make```

Then you can execute the program in virtual environment by:
```./venv/bin/python3 RNA3Dclust.py [options]```

### Usage
You can execute the program by:<br/>
```python3 RNA3Dclust.py -i infile -v -a M -o outfile  ```

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
  -i INPUT, --input INPUT
                        input file. Must be in pdb format.
  -c, --chain           process all chains at once. If not, the program will process each chain individually.
  -t THRESHOLD, --threshold THRESHOLD
                        Lower threshold for sequence length
  -o OUTFILE, --outfile OUTFILE
                        output file.
  -a {D,M,A,S,C}, --algorithm {D,M,A,S,C}
                        Clustering algorithm. Either: D (DBSCAN); M (MeanShift, default); A (Agglomerative); S (Spectral); C (Contact map-based))
```

- Each algorithm has its default parameters. For example, if you want to check the MeanShift, type ```python3 ./RNA3DClust.py M -h ``` for details. You can also change the parameters, in this case is the bandwidth (-b), by following: <br>
``` python3 RNA3Dclust.py -i infile -v -a M -o outfile M -b 5```

### Notes
- All parameters specified for each algorithm can only be accessed after typing its abbreviation (besides the option -a Algorithm);
- The input must be in pdb format;
- There are 2 output files if the output option is chosen. One file is the **JSON file**, which contains the coordinate, the residue number and the label of clusters. The other file contains the **command line for PyMOL GUI** to generate the clusters, which has the same name as the JSON file with the suffix '_pymolcmd.pml' (name ```outfile_pymolcmd.pml``` for example). You can either run it from terminal by this command:<br>
`pymol outfile_pymolcmd.pml`
<br/> if you already created alias for PyMOL. Or from PyMOL command line, you can try: <br/>
```@outfile_pymolcmd ```
