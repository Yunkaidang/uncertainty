import pandas as pd
import os

def update_csv_with_txt_folders(base_dir,txt_folder_paths, output_path, model_order_file):

    def process_txt_files(txt_folder_path):
        txt_folder_path=os.path.join(base_dir, txt_folder_path)
        txt_files = []
        for root, dirs, files in os.walk(txt_folder_path):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))

        all_data = []

        for txt_file in txt_files:
            model_name = os.path.basename(txt_file).replace('.txt', '')
            with open(txt_file, 'r') as file:
                lines = file.readlines()

            data = {
                'all': {'Consistent': 0, 'Inconsistent': 0, 'Consistent Ratio': 0, 'Inconsistent Ratio': 0},
                'T-T': {'Consistent': 0, 'Inconsistent': 0, 'Consistent Ratio': 0, 'Inconsistent Ratio': 0},
                'T-F': {'Consistent': 0, 'Inconsistent': 0, 'Consistent Ratio': 0, 'Inconsistent Ratio': 0},
                'F-F': {'Consistent': 0, 'Inconsistent': 0, 'Consistent Ratio': 0, 'Inconsistent Ratio': 0},
                'F-T': {'Consistent': 0, 'Inconsistent': 0, 'Consistent Ratio': 0, 'Inconsistent Ratio': 0},
            }

            consistent_count = 0
            inconsistent_count = 0
            current_label = None

            for line in lines:
                if 'Consistent Count:' in line:
                    consistent_count = int(line.split('Consistent Count: ')[1].split()[0])
                if 'Inconsistent Count:' in line:
                    inconsistent_count = int(line.split('Inconsistent Count: ')[1].split()[0])
                if 'Consistent Ratio:' in line:
                    if 'All data' in line:
                        current_label = 'all'
                    if 'True->True' in line:
                        current_label = 'T-T'
                    if 'True->False' in line:
                        current_label = 'T-F'
                    if 'False->False' in line:
                        current_label = 'F-F'
                    if 'False->True' in line:
                        current_label = 'F-T'
                    if current_label:
                        consistent_ratio = float(line.split('Consistent Ratio: ')[1].split('%')[0])
                        data[current_label]['Consistent Ratio'] = consistent_ratio
                if 'Inconsistent Ratio:' in line:
                    if 'All data' in line:
                        current_label = 'all'
                    if 'True->True' in line:
                        current_label = 'T-T'
                    if 'True->False' in line:
                        current_label = 'T-F'
                    if 'False->False' in line:
                        current_label = 'F-F'
                    if 'False->True' in line:
                        current_label = 'F-T'
                    if current_label:
                        inconsistent_ratio = float(line.split('Inconsistent Ratio: ')[1].split('%')[0])
                        data[current_label]['Inconsistent Ratio'] = inconsistent_ratio
                        data[current_label]['Consistent'] = consistent_count
                        data[current_label]['Inconsistent'] = inconsistent_count
                        current_label = None


            # df_new = pd.DataFrame({
            #     'model': [model_name],
            #     'all': [f"({data['all']['Consistent']}/{data['all']['Inconsistent']}){data['all']['Consistent Ratio']}%-{data['all']['Inconsistent Ratio']}%"],
            #     'T-T': [f"({data['T-T']['Consistent']}/{data['T-T']['Inconsistent']}){data['T-T']['Consistent Ratio']}%-{data['T-T']['Inconsistent Ratio']}%"],
            #     'T-F': [f"({data['T-F']['Consistent']}/{data['T-F']['Inconsistent']}){data['T-F']['Consistent Ratio']}%-{data['T-F']['Inconsistent Ratio']}%"],
            #     'F-F': [f"({data['F-F']['Consistent']}/{data['F-F']['Inconsistent']}){data['F-F']['Consistent Ratio']}%-{data['F-F']['Inconsistent Ratio']}%"],
            #     'F-T': [f"({data['F-T']['Consistent']}/{data['F-T']['Inconsistent']}){data['F-T']['Consistent Ratio']}%-{data['F-T']['Inconsistent Ratio']}%"]
            # })
            df_new = pd.DataFrame({
                'model': [model_name],
                'all': [f"{data['all']['Consistent Ratio']}%"],              
                'T-F': [f"{data['T-F']['Inconsistent Ratio']}%"],
                'F-T': [f"{data['F-T']['Consistent Ratio']}%"],

            })

            all_data.append(df_new)


        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            return df_combined
        else:
            return None


    model_order_df = pd.read_csv(model_order_file)

    for txt_folder_path in txt_folder_paths:
        df_combined = process_txt_files(txt_folder_path)
        if df_combined is not None:

            df_final = pd.merge(model_order_df, df_combined, on='model', how='left')

            csv_file_name = os.path.basename(txt_folder_path.rstrip('/\\')) + '.csv'
            csv_file_path = os.path.join(output_path, csv_file_name)

            df_final.to_csv(csv_file_path, index=False)
            print(f"Data from {txt_folder_path} written to {csv_file_path}")
        else:
            print(f"No valid data found in {txt_folder_path}, skipping...")

# Example usage
txt_folder_paths = [
    'test_dataset_6',
    ]
base_dir='../result/'
output_path = "Tables/"
model_order_file = 'model_order.csv'  
update_csv_with_txt_folders(base_dir,txt_folder_paths, output_path, model_order_file)
