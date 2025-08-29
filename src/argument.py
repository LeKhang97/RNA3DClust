import argparse
from pathlib import Path
from src.Functions import *

def main_argument():
    parser = argparse.ArgumentParser(description ="This tool is used to detect RNA domains from given 3D coordinate :))")

    subparsers = parser.add_subparsers(dest='subcommand')

    parser.add_argument('-v', '--verbose',
        action ='store_true', 
        help ='verbose mode.')
    
    parser.add_argument('-i',
        '--input',
        required=True,
        help ='input file. Must be in pdb format.')
    
    parser.add_argument('-at',
        '--atom_type',
        default = "C3'",
        help ="Atom types to be considered in the analysis. Default is C3'.")
    
    parser.add_argument('-t',
        '--threshold',
        default= 30,
        type= int,
        help ='Lower threshold for sequence length')

    parser.add_argument('-o', '--outpath',
                nargs='?', 
                const='.',
                default= None, 
                type=str,
                help ="path of output for json and pdb files. If not specified, the output will be saved in the current directory.")
    
    parser.add_argument('-c', '--chain',
                        type=str, 
                        nargs='?', 
                        const=False, 
                        default='all', 
                        help='Name of the chain to be processed. If not specified, all chains will be processed.')
    
    parser.add_argument('-j', '--json', 
                        type= str, 
                        nargs = '?', 
                        const = False, 
                        default = None, 
                        help='Name of the output json files. If not specified, its name will be the same as the input file')
    
    parser.add_argument('-p', '--pdb', 
                        type= str, 
                        nargs = '?', 
                        const = False, 
                        default = None, 
                        help='Name of the output pdb file(s). If not specified, its name will be the same as the input file')
    
    parser.add_argument('-a', 
					'--algorithm',
                    default = 'M',
					choices = ['D', 'M', 'A', 'S'],
					help="Clustering algorithm. Either: D (DBSCAN); M (MeanShift) (default); A (Agglomerative); S (Spectral)")
    
    '''parser.add_argument('-d',
                        '--dynamic',
                        action='store_true',
                        help='Use dynamic bandwidth for MeanShift clustering. Default is False.')'''
    
    parser.add_argument('-m', 
					'--mode',
                    default = 'static',
					choices = ['static', 'dynamic', 'adaptive'],
					help="Mode for MeanShift clustering. Either: static (default); dynamic; adaptive")
    
    # Subparser for -a D
    parser_a_D = subparsers.add_parser('D', help='Arguments for DBSCAN algorithm')
    parser_a_D.add_argument('-e', type=float, default= 0.5, help='espilon (default = 0.5)')
    parser_a_D.add_argument('-m', type=float, default = 5, help='min Pts (default = 5)')
    # Subparser for -a M
    parser_a_M = subparsers.add_parser('M', help='Arguments for MeanShift algorithm')
    parser_a_M.add_argument('-b', type=float, default= 0.2, help='bandwidth')
    parser_a_M.add_argument('-k', type= str, default= 'flat', choices = ['flat', 'gaussian'], help='kernel type (default = flat)') 

    # Subparser for -a A
    parser_a_A = subparsers.add_parser('A', help='Arguments for Agglomerative Clustering algorithm')
    parser_a_A.add_argument('-n', type=int, default= 2, help='number of clusters (default = 2)')
    parser_a_A.add_argument('-d', type=float, default= None, help='distance_threshold')

    # Subparser for -a S
    parser_a_S = subparsers.add_parser('S', help='Arguments for Spectral algorithm')
    parser_a_S.add_argument('-n', type=int, default= 2, help='number of clusters (default = 2)')
    parser_a_S.add_argument('-g', type=float, default= 1, help='gamma (default = 1)')

    # Subparser for -a C
    #parser_a_C = subparsers.add_parser('C', help='Arguments for Contact-based clustering algorithm')
    
    args = parser.parse_args()      
    
    return args

def process_args():
    args = main_argument()
    largs = [args.input, args.algorithm]

    algo_list = ['DBSCAN', 'MeanShift', 'Agglomerative', 'Spectral']
    
    algo = [i for i in algo_list if i[0] == args.algorithm][0]

    msg = 'Input information:'
    decorate_message(msg)

    print('Using atom type: ', args.atom_type)
    print("Using algorithm: ", algo)
    
    print(f'Mode selected for {algo} algorithm:', end = ' ')

    if args.outpath != None and args.json == None and args.pdb == None:
        args.json = args.input
        args.pdb = args.input
    
    if args.json == False:
        args.json = args.input

    if args.pdb == False:
        args.pdb = args.input
    
    if (args.outpath == None) and (args.json != None or args.pdb != None):
        args.outpath = '.'

    largs2 = [args.outpath, args.verbose, args.atom_type, args.json, args.pdb, args.chain]
        
    if args.algorithm == 'D':
        if not hasattr(args, 'e'):
            args.e = 0.5
            args.m = 5

        print(f"epsilon: {args.e}, min Pts: {args.m}")
        largs += [args.e, args.m]

    elif args.algorithm == 'M':
        if not hasattr(args, 'b'):
            args.b = 0.2
        
        if not hasattr(args, 'k'):
            args.k = 'flat'

        print(f"bandwidth: {args.b}, kernel type: {args.k}")
        largs += [args.b, args.k]

    elif args.algorithm == 'A':
        if not hasattr(args, 'n') and not hasattr(args, 'd'):
            args.n = 2
            args.d = None
        elif args.d != None:
            args.n = None
            if args.d < 0:
                sys.exit("Distance threshold must be a positive number!")

        print(args.n, args.d)
        if args.n == None:
            print(f"distance threshold: {args.d}")
            
        else:
            print(f"number of cluster: {args.n}")
            
        largs += [args.n, args.d]

    elif args.algorithm  == 'S':
        if not hasattr(args, 'n'):
            args.n = 2
            args.g = 1
        
        print(f"number of cluster: {args.n}, gamma: {args.g}")
        largs += [args.n, args.g]

    #elif args.algorithm == 'C':
    #    print("No arguments needed for Contact-based clustering")
            
    else:
        sys.exit("Unrecognized algorithm!")

    largs += [args.mode, args.threshold]
    print(largs)


    return largs, largs2
