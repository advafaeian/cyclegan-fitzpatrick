import pandas as pd
import shutil
import os
from tqdm import tqdm

base_dir = "./ddi_dataset/"

df = pd.read_csv(os.path.join(base_dir, "ddi_metadata.csv"))
ds_12 = df[df['skin_tone'] == 12]
ds_56 = df[df['skin_tone'] == 56]


base_dest_dir = os.path.join(base_dir, "train")

def copy_images(df, dest_folder):

    dest_dir = os.path.join(base_dest_dir, dest_folder)
    
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)

    for file_name in tqdm(df['DDI_file'], desc="Copying files"):

        source_path = os.path.join(base_dir, file_name)
        destination_path = os.path.join(dest_dir, file_name)
   
        if os.path.isfile(source_path):
            shutil.copy(source_path, destination_path)
        else:
            print(f"File not found: {file_name}")


copy_images(ds_12, "A")
copy_images(ds_56, "B")
