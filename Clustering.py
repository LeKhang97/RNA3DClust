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
            print(y[2])
            C = process_pdb(file)
            if '\\' in x[0]:
                filename = ''.join(x[0].split('\\')[-1]).replace('.pdb', '')
            else:
                filename = ''.join(x[0].split('/')[-1]).replace('.pdb', '')
            
            # Create a command file to PyMOL
            cmd_file = f'load {filename}.pdb; '

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
            sys.exit("No chain will be proccessed!")

        result = {filename:{}}

        #Check if the user want to cluster all chains together
        if y[2]:            
            pred = cluster_algo(flatten_np(data[:num_chains]), *x[1:])
            name = filename + '_chain_all_' + C[1][0].split('_')[1]
            pymol_cmd = pymol_proccess(pred, flatten_np(res_num_array[:num_chains]), name)
            print('\n')
            result[filename]['chain_all'] = {'data': flatten_np(data[:num_chains]),
                                            'cluster': pred,
                                            'res': flatten_np(res_num_array[:num_chains]),
                                            'PyMOL': pymol_cmd
                                            }
            
            cmd_file += '; '.join(pymol_cmd)
                        
        #If the user want to cluster each chain separately
        else:
            for subdata, res_num, i in zip(data, res_num_array, C[1]):
                pred = cluster_algo(subdata, *x[1:])

                name = filename + f'_chain_{i}'
                pymol_cmd = pymol_proccess(pred, res_num, name)
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
        if y[1]:
            print(f"Writing to {y[0]}", end='\n\n')
            print(f"Writing to {y[0]}_pymolcmd", end='\n\n')

        if y[0].split('.')[-1] != 'json':
            outfile1 = y[0] + '.json'
            outfile2 = y[0] + '_pymolcmd'
        else:
            outfile1 = y[0]
            outfile2 = y[0].replace('.json', '_pymolcmd')
        
        with open(outfile1, 'w') as outfile:
            json.dump(result, outfile, indent=2, cls=NumpyEncoder)
        
        with open(outfile2, 'w') as outfile:
            outfile.write(cmd_file)
        
        if y[1]:
            print("Writing completed!")