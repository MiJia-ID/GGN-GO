import json
import argparse
from Predictor import Predictor

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fasta_file', type=str, default='../Predict/3gdt-a.fasta', help="Fasta file with protein sequences.")
    parser.add_argument('--task', type=str, default='mf', choices=['bp', 'mf', 'cc'], help='')
    parser.add_argument('--model_path', type=str, default='../Model/model_mf.pt', help='')


    parser.add_argument('-o', '--output_fn_prefix', type=str, default='../Predict/', help="Save predictions/saliency in file.")
    parser.add_argument('-v', '--verbose', help="Prints predictions.", action="store_true")
    parser.add_argument('--use_guided_grads', default=True,help="Use guided grads to compute gradCAM.", action="store_true")
    parser.add_argument('--saliency', default='mf',help="Compute saliency maps for every protein and every MF-GO term/EC number.", action="store_true")

    parser.add_argument("--prottrans_output_path", type=str, default='../data/bpDataSet/prottrans/')
    parser.add_argument("--esm2_output_path", type=str, default='../data/bpDataSet/esm2/')
    parser.add_argument('--pdb_dir', type=str, default='../data/pdb_dir/')
    parser.add_argument("--proc_pdb_output_path", type=str, default='../data/bpDataSet/structure_data/')
    parser.add_argument("--dssp_output_path", type=str, default='../data/bpDataSet/dssp/')
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument('--pooling', default='MTP', type=str, choices=['MTP', 'GMP'],
                        help='Multi-set transformer pooling or Global max pooling')
    parser.add_argument('--device', type=str, default='3', help='')
    parser.add_argument('--contrast', default=True, type=str2bool, help='whether to do contrastive learning')
    args = parser.parse_args()


    predictor = Predictor(args.model_path)
    predictor.predict(args.fasta_file,args.task,args.pooling,args.contrast,args.device,args)

    # save predictions
    predictor.export_csv(args.output_fn_prefix + args.task.upper() + "_predictions.csv", args.verbose)
    predictor.save_predictions(args.output_fn_prefix + args.task.upper() + "_pred_scores.json")

    #save saliency maps
    if args.saliency and args.task in ['mf', 'ec']:
        predictor.compute_GradCAM( use_guided_grads=args.use_guided_grads)
        predictor.save_GradCAM(args.output_fn_prefix + "_" + args.task.upper() + "_saliency_maps.json")
