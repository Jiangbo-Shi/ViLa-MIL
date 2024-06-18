import os
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str)
parser.add_argument('--val_frac', type=float, default= 0.2,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.2,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--dataset', type=str, default='Yifuyuan')

args = parser.parse_args()

if args.task == 'task_tcga_rcc_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/TCGA_RCC_subtyping.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'CCRCC':0, 'PRCC':1, 'CRCC':2},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])

elif args.task == 'task_tcga_lung_subtyping':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/TCGA_Lung_subtyping.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'LUAD':0, 'LUSC':1},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])
else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ args.dataset +'/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



