import pandas as pd 
import numpy as np 
import os 

N = 'SHOT_NUMBER'
data_folder = 'DATA_FOLDER'
all_data = np.array(pd.read_excel('ALL_DATA_PATH', engine='openpyxl',  header=None))
save_folder = 'SAVE_FOLDER'

if(not os.path.exists(save_folder)):
    os.makedirs(save_folder)

for j in range(5):
    orginal_data_split_path = data_folder + '/splits_'+str(j)+'.csv'
    orginal_data_stastic_path = data_folder + '/splits_'+str(j)+'_descriptor.csv'
    save_path = save_folder + '/splits_'+str(j)+'.csv'

    orginal_data_split = np.array(pd.read_csv(orginal_data_split_path))
    slidename2label = {}
    for each_data in all_data:        
    all_slide_label = []
    selected_train_slide = []
    for each_data in orginal_data_split:
            slide_label = slidename2label[each_data[1]]
            all_slide_label.append(slide_label)
    unique_label = np.unique(all_slide_label)
    for each_label in unique_label:
        each_index = np.where(all_slide_label == each_label)[0]
        selected_index = np.random.choice(each_index, size=N, replace=False)
        for each_index in selected_index:
            selected_train_slide.append(orginal_data_split[each_index][1])


    orginal_data_split[:, 1][0:len(selected_train_slide)] = selected_train_slide
    orginal_data_split[:, 1][len(selected_train_slide):-1] = np.nan

    all_nums = np.array(pd.read_csv(orginal_data_stastic_path))
    val_num = np.sum(all_nums[:, 2])

    new_data_split = orginal_data_split[:val_num]

    column_name = ['','train', 'val', 'test']
    csv = pd.DataFrame(columns=column_name, data = new_data_split)
    csv.to_csv(save_path, index=False)
