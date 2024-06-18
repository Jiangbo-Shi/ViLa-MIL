from __future__ import print_function

import numpy as np
import argparse
import torch
import os
import pandas as pd
from utils.utils import *
from math import floor
from datasets.dataset_generic import Generic_MIL_Dataset
from utils.eval_utils import *

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation settings 
parser = argparse.ArgumentParser(description='Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
parser.add_argument('--data_folder_s', type=str, default=None, help='dir under data directory' )
parser.add_argument('--data_folder_l', type=str, default=None, help='dir under data directory' )
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['ViLa_MIL'], default='ViLa_MIL')
parser.add_argument('--mode', type=str, choices=['transformer'], default='transformer')
parser.add_argument('--drop_out', action='store_true', default=False, help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str)
parser.add_argument("--text_prompt", type=str, default=None)
parser.add_argument("--text_prompt_path", type=str, default=None)

args = parser.parse_args()
args.text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze()
args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'mode': args.mode,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)

if args.task == 'task_tcga_rcc_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/TCGA_RCC_subtyping.csv',
                                  mode = args.mode,
                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'CCRCC':0, 'PRCC':1, 'CRCC':2},
                                  patient_strat= False,
                                  ignore=[])

elif args.task == 'task_tcga_lung_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/TCGA_Lung_subtyping.csv',
                                  mode = args.mode,
                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'LUAD':0, 'LUSC':1},
                                  patient_strat= False,
                                  ignore=[])
else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    all_f1 = []
    all_true = []
    all_pred = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, patient_results, test_error, auc, test_f1, df, _ = eval(args.mode, split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        all_f1.append(test_f1)

        all_true += list(df['Y'])
        all_pred += list(df['Y_hat'])

        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc,'test_acc': all_acc, 'test_f1': all_f1})
    result_df = pd.DataFrame({'metric': ['mean', 'var'],
                              'test_auc': [np.mean(all_auc), np.std(all_auc)],
                              'test_acc': [np.mean(all_acc), np.std(all_acc)],
                              'test_f1': [np.mean(all_f1), np.std(all_f1)],
                              })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
        result_name = 'result_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
        result_name = 'result.csv'

    result_df.to_csv(os.path.join(args.save_dir, result_name), index=False)
    final_df.to_csv(os.path.join(args.save_dir, save_name))
