import argparse
from pathlib import Path
from Functions import *

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
    
    parser.add_argument('-c',
        '--chain',
        action='store_true',
        help ='process all chains at once. If not, the program will process each chain individually.')
    
    parser.add_argument('-t',
        '--threshold',
        default= 30,
        type= int,
        help ='Lower threshold for sequence length')

    parser.add_argument('-o', '--outfile',
                #default = None,
                action ='store',
                help ='output file.')
    
    parser.add_argument('-a', 
					'--algorithm',
                    default = 'M',
					choices = ['D', 'M', 'A', 'S'],
					help="Clustering algorithm. Either: D (DBSCAN); M (MeanShift) (default); A (Agglomerative); S (Spectral))")
    
    # Subparser for -a D
    parser_a_D = subparsers.add_parser('D', help='Arguments for DBSCAN algorithm')
    parser_a_D.add_argument('-e', type=float, default= 0.5, help='espilon (default = 0.5)')
    parser_a_D.add_argument('-m', type=int, default = 5, help='min samples (default = 5)')

    # Subparser for -a M
    parser_a_M = subparsers.add_parser('M', help='Arguments for MeanShift algorithm')
    parser_a_M.add_argument('-b', type=float, default= 0.2, help='bandwidth')
    parser_a_M.add_argument('-k', type= str, default= 'flat', choices = ['flat', 'gaussian'], help='kernel type (default = flat)') 
    parser_a_M.add_argument('-a', type= str, default= 'False', choices = ['True', 'False'], help= 'recalculate bandwidth after each iteration (default = False)')

    # Subparser for -a A
    parser_a_A = subparsers.add_parser('A', help='Arguments for Agglomerative Clustering algorithm')
    parser_a_A.add_argument('-n', type=int, default= 2, help='number of clusters (default = 2)')
    parser_a_A.add_argument('-d', type=int, default= None, help='distance_threshold')

    # Subparser for -a S
    parser_a_S = subparsers.add_parser('S', help='Arguments for Spectral algorithm')
    parser_a_S.add_argument('-n', type=int, default= 2, help='number of clusters (default = 2)')
    parser_a_S.add_argument('-g', type=float, default= 1, help='gamma (default = 1)')

    args = parser.parse_args()  
    
    return args

def process_args():
    args = main_argument()
    largs = [args.input, args.algorithm]
    largs2 = [args.outfile, args.verbose, args.chain]

    algo_list = ['DBSCAN', 'MeanShift', 'Agglomerative', 'Spectral']
    
    algo = [i for i in algo_list if i[0] == args.algorithm][0]

    if args.verbose:
        print("Using algorithm: ", algo)
        print(f"Arguments for {algo}:")
        
    if args.algorithm == 'D':
        if not hasattr(args, 'e'):
            args.e = 0.5
            args.m = 5

        if args.verbose:
            print(f"e: {args.e}, m: {args.m}")
        largs += [args.e, args.m]

    elif args.algorithm == 'M':
        if not hasattr(args, 'b'):
            args.b = 0.2
        
        if not hasattr(args, 'k'):
            args.k = 'flat'
        if not hasattr(args, 'a'):
            args.a = False

        elif args.a == 'True':
            args.a = True
        
        elif args.a == 'False':
            args.a = False

        if args.verbose:
            print(f"b: {args.b}, k: {args.k}, a: {args.a}")
        largs += [args.b, args.k, args.a]

    elif args.algorithm == 'A':
        if not hasattr(args, 'n') and not hasattr(args, 'd'):
            args.n = 2
            args.d = None
        if hasattr(args, 'd'):
            args.n = None

        if args.verbose:
            print(f"n: {args.n}")
        largs += [args.n]
        
    elif args.algorithm  == 'S':
        if not hasattr(args, 'n'):
            args.n = 2
            args.g = 1
        
        if args.verbose:
            print(f"n: {args.n}, g: {args.g}")
        largs += [args.n, args.g]

    else:
        sys.exit("Unrecognized algorithm!")

    largs += [args.threshold]

    return largs, largs2