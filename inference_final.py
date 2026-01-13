import os
import glob
import shutil
import gc
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
# Ensure functions.py is importable. It's in the same directory.
from functions import color_correction, read_QR_code, get_all_features_parallel, save_ROI_parallel, get_ids

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='HopBox Inference Script')
    
    # Paths
    parser.add_argument('--input-dir', type=str, required=True, help='Path to input directory containing images')
    parser.add_argument('--model', type=str, default='/app/models/best.pt', help='Path to .pt model file (default: built-in model)')
    parser.add_argument('--output-dir', type=str, default=None, help='Path to save results (default: input_dir + "_Results")')
    
    # Inference settings
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--imgsz', type=int, default=2048, help='Inference image size')
    parser.add_argument('--conf', type=float, default=0.58, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.75, help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='', help='Device to run on (e.g., 0, 0,1,2,3 or cpu). Leave empty for auto.')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    gc.enable()
    gc.collect()

    # Determine device
    if args.device:
        device = args.device
    else:
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    if device == '0':
        torch.cuda.empty_cache()
    
    print(f'Using device: {device}')

    # --- Configuration ---
    model_dir = args.model
    Data_folder = args.input_dir
    
    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"Error: Model not found at {model_dir}")
        print(f"Current working directory: {os.getcwd()}")
        return

    # Check if Data_folder exists
    if not os.path.exists(Data_folder):
        print(f"Error: Data folder not found at {Data_folder}")
        return

    save_dir = args.output_dir if args.output_dir else Data_folder + '_Results'
    img_fmt = ".JPG"
    nx = args.batch_size

    model_params = {
        'project': 'HopBox',
        'name': 'Results',
        'save': True, 
        'batch': -1, 
        'show_labels': True,
        'show_conf': True,
        'save_crop': True, 
        'line_width': 3, 
        'conf': args.conf,
        'iou': args.iou,
        'imgsz': args.imgsz,
        'exist_ok': True, 
        'half': True, # Use FP16 half-precision inference
        'cache': False, 
        'retina_masks': True, 
        'device': device 
    }
    
    # Color correlation parameters (kwargs)
    cc_kwargs = {
        'method': 'pls', 
        'degree': 3, 
        'interactions_only': True,
        'ncomp': 12,
        'max_iter': 5000, 
        'white_balance_mtd': 3,
    }

    # --- Processing ---
    
    # Get subfolders
    folder_list = [os.path.join(Data_folder, f) for f in os.listdir(Data_folder) if os.path.isdir(os.path.join(Data_folder, f))]
    # remove folders that have '_complete' in them (or are hidden)
    folder_list = [f for f in folder_list if '_complete' not in f and not os.path.basename(f).startswith('.')]

    base_folder_name = os.path.basename(Data_folder)
    print(f'{len(folder_list)} folders found in Folder `{base_folder_name}`')

    try:
        model = YOLO(model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    for image_dir in folder_list:
        folder_bname = os.path.basename(image_dir)
        print(f'Processing folder: {folder_bname.upper()}')
        
        save_dir_folder = os.path.join(save_dir, folder_bname)
        if not os.path.exists(save_dir_folder):
            os.makedirs(save_dir_folder)

        # Update model params output dir
        model_params['project'] = save_dir_folder
        
        # Get images
        # Match .JPG case-insensitive if possible, but glob is case-sensitive on Linux, Windows usually case-insensitive.
        # The notebook used: os.path.join(image_dir, '**/*'+img_fmt) with recursive=True.
        img_paths = glob.glob(os.path.join(image_dir, '**/*' + img_fmt), recursive=True)
        num_imgs = len(img_paths)
        print(f"Number of images found = {str(num_imgs)}")
        
        if num_imgs == 0:
            print(f"No images found in {folder_bname}, skipping...")
            continue
            
        # Batches
        img_paths_batches = [img_paths[i:i + nx] for i in range(0, len(img_paths), nx)]
        
        if device == "0":
            print("Clearing CUDA memory")
            torch.cuda.empty_cache()
            gc.collect()

        print('Running inference...')
        
        # We need to set name for each folder iteration 
        current_name_param = 'Results_' + folder_bname
        model_params['name'] = current_name_param

        DF_all = []
        
        for iz, batch_paths in enumerate(img_paths_batches):
            gc.collect()
            if device == '0':
                torch.cuda.empty_cache()
            
            print(f'Running for Batch #{iz+1}/{len(img_paths_batches)} with {len(batch_paths)} images')
            
            # Predict
            try:
                # YOLO predict can take a list of paths
                results = model.predict(source=batch_paths, **model_params)
            except Exception as e:
                print(f"Error during model prediction: {e}")
                continue

            print('2. Extracting features from results...')
            
            DF_batch = pd.DataFrame()
            
            for i, result in enumerate(results):
                try:
                    img_path = result.path
                    img_name = os.path.basename(img_path)
                    print(f'Working on image: {img_name}')
                    
                    result = result.cpu()
                    # Color correction
                    result, patch_size = color_correction(result, kwargs=cc_kwargs)
                    
                    try:
                        QR_info = read_QR_code(result)
                    except Exception as e:
                        print(f'Could NOT read QR code info. for image "{img_name}": {e}')
                        QR_info = 'NA'
                    
                    df = get_all_features_parallel(result, name='Hops')
                    
                    # Create DataFrame rows
                    w, _ = df.shape
                    img_name_col = [img_name] * w
                    QR_info_col = [QR_info] * w
                    patch_size_col = [np.mean(patch_size)] * w
                    indeces = list(range(w))
                    
                    df_fore = pd.DataFrame({
                        'Image_name': img_name_col,
                        'ID': indeces,
                        'QR_info': QR_info_col,
                        'Patch_size': patch_size_col
                    })
                    
                    df = pd.concat([df_fore, df], axis=1)
                    DF_batch = pd.concat([DF_batch, df], axis=0, ignore_index=True)
                    
                    # Save crops
                    img_save_folder = os.path.join(save_dir_folder, 'Image_Predictions')
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder)
                        
                    save_ROI_parallel(result, get_ids(result, 'Hops'), os.path.join(img_save_folder, img_name))
                    
                    print(f"Image {i+1} of {len(results)} processed.")
                    
                    del result, df
                    gc.collect()
                    if device == '0':
                         torch.cuda.empty_cache()

                except Exception as e:
                    print(f'Could NOT process image "{img_name}"... Skipping image. Error: {e}')
                    # import traceback
                    # traceback.print_exc()
                    pass
            
            DF_all.append(DF_batch)
            del DF_batch, results
            gc.collect()
            if device == '0':
                torch.cuda.empty_cache()
                
        try:
            if DF_all:
                final_df = pd.concat(DF_all, axis=0, ignore_index=True)
                csv_path = os.path.join(save_dir_folder, 'Features.csv')
                final_df.to_csv(csv_path, index=False)
                print(f"Features saved to {csv_path}")
            else:
                print("No features extracted.")
        except Exception as e:
            print(f"Error saving CSV: {e}")
            pass
            
        print(f"Moving processed folder {image_dir} to {image_dir}_complete")
        try:
             shutil.move(image_dir, image_dir + '_complete')
        except Exception as e:
            print(f"Error moving folder: {e}")

    print('Done.')

if __name__ == '__main__':
    main()
