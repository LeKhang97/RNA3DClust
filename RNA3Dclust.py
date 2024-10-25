from argument import *

if __name__ == "__main__":
    x, y  = process_args()  #Check argument.py for more details
        
    #Check if file or file path exists
    if Path(x[0]).exists() == False:
        sys.exit("Filename does not exist!")
    else:
        if y[1]:
            print(f"\nProcessing {x[0]}", end='\n\n')

        #Read file
        with open(x[0], 'r') as infile:
            file = infile.read().split('\n')
            C = process_pdb(file, atom_type=y[2])
            if C == False:
                sys.exit("File is error or chosen atom type is not present!")

            if '\\' in x[0]:
                filename = ''.join(x[0].split('\\')[-1]).replace('.pdb', '')
            else:
                filename = ''.join(x[0].split('/')[-1]).replace('.pdb', '')
            
            # Create a command file to PyMOL
            #cmd_file = f'load {os.getcwd()}\{filename}.pdb; '
            cmd_file = f'load {os.getcwd()}\{x[0]}; '
            #print(cmd_file, f'load {os.getcwd()}\{x[0]}; ')

    #Check if the file is error, the result is either empty or if the length of chains is valid
    data, res_num_array  = check_C(C, x[-1])
    if data ==  False:
        sys.exit("File is error!")
    else:
        models = set([''.join(i.split('_')[1]) for i in C[1]])
        if bool(models):
            num_chains = len([i for i in C[1] if ''.join(i.split('_')[1]) == list(models)[0]])
        else:
            num_chains = len(C[1])
        
        if y[1]:
            print("Number of models: ", len(models))
            print("Models: ", models, end='\n\n')
            print("Number of chains: ", num_chains, end='\n\n')
        
        # If the length of all chains is invalid, the program will exit
        if len(data) == 0:
            sys.exit("No chain will be processed!")

        result = {filename:{}} 
                        
        # Cluster each chain separately
        for subdata, res_num, i in zip(data, res_num_array, C[1]):
            pred = cluster_algo(subdata, *x[1:])
            print(set(pred))
            if x[1][0] != 'C':
                if y[1]:
                    if -1 in pred:
                        print('Noise points detected before post-processing!')
                pred = post_process(pred, res_num)
                if y[1]:
                    if -1 in pred:
                        print('Noise points detected after post-processing!')
            else:
                flatten_pred = [i for j in pred for i in j]
                pred = [c for i in flatten_pred for c in range(len(pred)) if i in pred[c]]

            name = filename + f'_chain_{i}'
            pymol_cmd = pymol_process(pred, res_num, name)
            print('\n')
            result[filename][f'chain_{i}'] = {'data': subdata,
                                            'cluster': pred,
                                            'res': res_num,
                                            'PyMOL': pymol_cmd
                                            }
            if i.split('_')[1] == 'MODEL1' or 'MODEL' not in i:
                cmd_file += '; '.join(pymol_cmd) + ';'
                
    #Check if the user want to write the result to a file
    if y[0] != None:
        target_dir = Path(y[0])
        if y[1]:
            print(f"Writing to the path {target_dir}", end='\n\n')
        
        target_dir.mkdir(parents=True, exist_ok=True)

        if y[3] != None:
            basename1 = os.path.basename(y[3])

            outfile1 = target_dir / f"{basename1.replace('.json','').replace('.pdb','')}.json"
            outfile2 = target_dir / f"{basename1.replace('.json','').replace('.pdb','')}_pymolcmd.pml"
            
            with open(outfile1, 'w') as outfile:
                json.dump(result, outfile, indent=2, cls=CustomNumpyEncoder)
        
            with open(outfile2, 'w') as outfile:
                outfile.write(cmd_file)
            
            if y[1]:
                print(f"Wrote {outfile1} and {outfile2}")

        if y[4] != None:
            basename2 = os.path.basename(y[4])
            name = x[0].replace('.pdb', '')
            for chain in result[filename].keys():
                pred = result[filename][chain]['cluster']
                res_num = result[filename][chain]['res']

                res = [res_num[i] for i in range(len(pred)) if pred[i] != -1]
                pred = [i for i in pred if i != -1]

                cluster_result = process_cluster_format(pred, res_num)
                cluster_lines = split_pdb_by_clusters(x[0], cluster_result, name, i.replace('_',''))
                # Write the output files for each cluster

                for cluster_index, cluster in cluster_lines.items():
                    if cluster:
                        output_file = target_dir / f"{basename2.replace('.pdb', '')}_{chain}cluster_{cluster_index + 1}.pdb"
                        with open(output_file, 'w') as outfile:
                            outfile.writelines(cluster)
                        if y[1]:
                            print(f"Wrote {len(cluster)} lines to {output_file}")
        
        if y[1]:
            print("Writing completed!")
