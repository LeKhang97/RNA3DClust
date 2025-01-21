![Version](https://img.shields.io/badge/Version-1.0.0-blue)
[![Platform](https://img.shields.io/badge/Platform-Linux-blueviolet)](https://evryrna.ibisc.univ-evry.fr/evryrna/RNA3DClust/home)
[![Docker supported](https://img.shields.io/badge/Docker-Supported-brightgreen)](https://github.com/LeKhang97/RNA3DClust/blob/main/Dockerfile)
[![GitHub repo](https://img.shields.io/badge/Repo-GitHub-white.svg)](https://github.com/LeKhang97/RNA3DClust/tree/main)
[![License](https://img.shields.io/badge/License-MIT-informational.svg)](https://github.com/LeKhang97/RNA3DClust/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2025.01.12.632579-yellow)](https://doi.org/10.1101/2025.01.12.632579)

# RNA3DClust: unsupervised segmentation of RNA 3D structure using density-based clustering
RNA3DClust uses a clustering-based approach to segment an input RNA 3D structure into independent regions, reminiscent of structural domains in proteins. Among the various options described below, RNA3DClust offers the possibility to generate PDB files, one for each domain found. It can also provide [PyMOL >= 2.5](https://www.pymol.org/) commands for coloring the delimited domains.

![Preview](https://evryrna2.ibisc.univ-evry.fr/RNA3DClust.png)

### Installation
There are two ways to install RNA3DClust:

1. Using Docker:  
```docker pull lequockhang/rna3dclust ```  
Then use this command below to run the docker image:  
```docker run -v `pwd`:/workdir/test lequockhang/rna3dclust [options] ```  
where `` `pwd` `` is your full path to your working directory containing input file(s).  
Here is the full command for example:  
```docker run -v `pwd`:/workdir/test lequockhang/rna3dclust -i Example.pdb -a A -v -p Output_pdb_name -j Output_json_name```

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
```python3 RNA3Dclust.py -i infile -v -a M -o path_for_output ```

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

  -at, --atom_type      Atom types to be considered in the analysis. The default is C3.

  -t THRESHOLD, --threshold THRESHOLD
                        Lower threshold for sequence length

  -o OUTPATH, --outpath OUTPATH
                        output path for json and pdb files. If not specified, the output will be saved in the current directory.

  -p PDB, --pdb PDB
                        output filename in PDB format.

  -j JSON, --json JSON
                        output filename in JSON format.

  -a {D,M,A,S,C}, --algorithm {D,M,A,S,C}
                        Clustering algorithm. Either: D (DBSCAN); M (MeanShift, default); A (Agglomerative); S (Spectral); C (Contact map-based))
```

- Each algorithm has its default parameters. For example, if you want to check the MeanShift, type ```python3 ./RNA3DClust.py M -h ``` for details. You can also change the parameters, in this case is the bandwidth (-b), by following: <br>
```python3 RNA3Dclust.py -i infile -v -a M -o . M -b 5```

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
