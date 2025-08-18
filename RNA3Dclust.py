from src.argument import *

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
            missing_res = find_missing_res(x[0])

            file = infile.read().split('\n')
            C = process_structure(file, atom_type=y[2], filename=x[0])
            #C = process_pdb(file, atom_type=y[2])

            if C == False:
                sys.exit("File is error or chosen atom type is not present!")

            if '\\' in x[0]:
                filename = ''.join(x[0].split('\\')[-1]).replace('.pdb', '').replace('.cif','')
            else:
                filename = ''.join(x[0].split('/')[-1]).replace('.pdb', '').replace('.cif','')
            
            # Create a command file to PyMOL
            #cmd_file = f'load {os.getcwd()}\{filename}.pdb; '
            cmd_file = f'load {os.getcwd()}/{x[0]}; '
            #print(cmd_file, f'load {os.getcwd()}\{x[0]}; ')

    #Check if the file is error, the result is either empty or if the length of chains is valid
    data, res_num_array, removed_chain_index  = check_C(C, x[-1])

    remaining_chains = [C[1][i] for i in range(len(C[1])) if i not in removed_chain_index]

    if data ==  False:
        sys.exit("File is error!")
    else:
        models = ''
        #models = set([''.join(i.split('_')[1]) for i in remaining_chains])
        if bool(models):
            num_chains = len([i for i in remaining_chains if ''.join(i.split('_')[1]) == list(models)[0]])
        else:
            num_chains = len(remaining_chains)
        
        if y[1]:
            print("Number of models: ", len(models))
            if models == set(['']):
                print("Model's name has not been specified")
            else:
                print("Models: ", sorted(models), end='\n\n')
            print("Number of chains: ", num_chains, end='\n\n')
        
        # If the length of all chains is invalid, the program will exit
        if len(data) == 0:
            sys.exit("No chain will be processed!") 

        result = {filename:{}} 
                        
        # Cluster each chain separately
        old_model = ''
        if y[5] == 'longest':
            print(f'Processing longest valid chain!', end='\n\n')
            chains = [i for i in remaining_chains if len(data[remaining_chains.index(i)]) == max([len(data[j]) for j in range(len(data))])]
        elif y[5] == 'all':
            print(f'Processing all valid chains!', end='\n\n')
            chains = remaining_chains
        elif y[5]:
            print(f'Processing chain {y[5]}!', end='\n\n')
            chains = [i for i in remaining_chains if i == y[5]]
            data = [data[remaining_chains.index(i)] for i in chains]
            res_num_array = [res_num_array[remaining_chains.index(i)] for i in chains]
        else:
            raise ValueError("Cannot find the chain to process, please check your chain name!")

        for subdata, res_num, i in zip(data, res_num_array, chains):
            if i in missing_res:
                missing_res_msg = ''.join([f'{l[0]}-{l[-1]}; ' if l[0] != l[-1] else f'{l[0]}; ' for l in list_to_range(missing_res[i])])
                print(f"Missing residues on chain {i}: {missing_res_msg}", end='\n\n')
            else:
                print(f"No missing residues on chain {i}", end='\n\n')

            if 'MODEL' in i:
                chain = i.split('_')[0]
                model = i.split('_')[1]

            else:
                model = ''
                chain = i.replace('_','')
            
            if model != old_model:
                print(model)
                old_model = model

            name = filename + f'_chain_{chain}'
            pred = cluster_algo(subdata, *x[1:], chain)
            
            if True:
                if x[1][0] != 'C':
                    pred = post_process_update(pred, res_num)
                else:
                    flatten_pred = [i for j in pred for i in j]
                    pred = [c for i in flatten_pred for c in range(len(pred)) if i in pred[c]]

            
            num_clusters = len(set(i for i in pred if i != -1))
            outlier = 'with' if -1 in pred else 'without'

            # Print the number of clusters and the presence of outliers
            msg = 'Output information:'
            decorate_message(msg)
            print(f'Chain {chain} has {num_clusters} clusters and {outlier} outliers.')

            pred_ext, res_num_ext = extend_missing_res(pred, res_num)
            #print(res_num)

            # Use extended data or not
            use_pred = pred_ext; use_res_num = res_num_ext

            # Print the number of residues in each cluster and their positions
            for h,k in enumerate(set(j for j in use_pred if j != -1)):
                range_pos = list_to_range([use_res_num[j] for j in range(len(use_pred)) if use_pred[j] == k])
                mess_pos = ''.join(f'{j[0]}-{j[-1]}, ' for j in range_pos)
                print(f'Number of residues of cluster {h+1}: {len([j for j in use_pred if j == k])}')
                print(f'Cluster {h+1} positions:\n{mess_pos}\n')
            
            # If there are outliers, print the number of outliers and their positions
            if -1 in use_pred:
                range_pos = list_to_range([use_res_num[j] for j in range(len(use_pred)) if use_pred[j] == -1])
                mess_pos = ''.join(f'{j[0]}-{j[-1]}, ' for j in range_pos)
                print(f'Number of residues of outliers: {len([j for j in use_pred if j == -1])}')
                print(f'Outliers positions:\n{mess_pos}\n')

            pymol_cmd = pymol_process(use_pred, use_res_num, name, verbose = y[1])

            print('\n')
            result[filename][i.replace('_','')] = {'data': subdata,
                                            'cluster': pred,
                                            'res': res_num,
                                            'missing_res': missing_res[i] if i in missing_res else None,
                                            'PyMOL': pymol_cmd
                                            }
            cmd_file += '; '.join(pymol_cmd) + ';'
                
    #Check if the user want to write the result to a file
    if y[0] != None:
        target_dir = Path(y[0])
        if y[1]:
            msg = 'Exporting output:'
            decorate_message(msg)
            print(f"Writing to the path {target_dir}", end='\n\n')
        
        target_dir.mkdir(parents=True, exist_ok=True)

        if y[3] != None:
            basename1 = os.path.basename(y[3])

            outfile1 = target_dir / f"{basename1.replace('.json','').replace('.pdb','').replace('.cif','')}.json"
            outfile2 = target_dir / f"{basename1.replace('.json','').replace('.pdb','').replace('.cif','')}_pymolcmd.pml"
            
            with open(outfile1, 'w') as outfile:
                json.dump(result, outfile, indent=2, cls=CustomNumpyEncoder)
        
            with open(outfile2, 'w') as outfile:
                outfile.write(cmd_file)
            
            if y[1]:
                print(f"Wrote {outfile1} and {outfile2}")

        if y[4] != None:
            basename2 = os.path.basename(y[4])
            name = x[0].replace('.pdb', '').replace('.cif','')
            for chain in result[filename].keys():
                pred = result[filename][chain]['cluster']
                res_num = result[filename][chain]['res']

                #res = [res_num[i] for i in range(len(pred)) if pred[i] != -1]
                #pred = [i for i in pred if i != -1]
                res = list(res_num) 

                cluster_result = process_cluster_format(pred, res)
                cluster_lines = split_pdb_by_clusters(x[0], cluster_result, name, chain)
                # Write the output files for each chain it processed
                if y[1]:
                    print(f"Writing clusters for chain {chain}...")
                
            
                output_file = target_dir / f"{filename}.pdb"

                with open(output_file, 'w') as outfile:
                    #print(data[remaining_chains.index(chain.replace('chain_',''))])
                    #subdata = '\n'.join(data[remaining_chains.index(chain.replace('chain_',''))])
                    outfile.writelines([i for j in cluster_lines.keys() for i in cluster_lines[j]])

                '''for cluster_index, cluster in cluster_lines.items():
                    if cluster:
                        output_file = target_dir / f"{basename2.replace('.pdb', '')}_{chain}cluster_{cluster_index + 1}.pdb"
                        with open(output_file, 'w') as outfile:
                            outfile.writelines(cluster)
                        if y[1]:
                            print(f"Wrote {len(cluster)} lines to {output_file}")'''
        
        if y[1]:
            print("Writing completed!")
