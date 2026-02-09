import pandas as pd
import os

def filename_no_ext(path):
    return os.path.splitext(os.path.basename(str(path)))[0]

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    df.columns = [c.replace(' ', '_') for c in df.columns]
    if 'blurred_psnr' not in df.columns: df['blurred_psnr'] = 0
    if 'blurred_ssim' not in df.columns: df['blurred_ssim'] = 0

    if 'algorithm' in df.columns:
        if 'psnr_improvement' not in df.columns and 'psnr' in df.columns:
            df['psnr_improvement'] = df['psnr'] - df['blurred_psnr']

        if 'ssim_improvement' not in df.columns and 'ssim' in df.columns:
            df['ssim_improvement'] = df['ssim'] - df['blurred_ssim']

        df['test_case_id'] = df.apply(
            lambda x: f"{filename_no_ext(x.get('image', ''))} ({filename_no_ext(x.get('filter', ''))})",
            axis=1
        )
        return df

    psnr_col = next((c for c in df.columns if c.startswith('psnr_') and c != 'blurred_psnr' and c != 'psnr_improvement'), None)
    
    if psnr_col:
        algo_name = psnr_col.replace('psnr_', '')
        
        new_df = pd.DataFrame()
        new_df['image'] = df['original'].apply(filename_no_ext)
        new_df['filter'] = df['kernel_blur'].apply(filename_no_ext)
        new_df['algorithm'] = algo_name
        
        new_df['psnr'] = df[f'psnr_{algo_name}']
        new_df['blurred_psnr'] = df['blurred_psnr']
        new_df['psnr_improvement'] = new_df['psnr'] - new_df['blurred_psnr']

        ssim_col = f'ssim_{algo_name}'
        if ssim_col in df.columns:
            new_df['ssim'] = df[ssim_col]
            new_df['blurred_ssim'] = df.get('blurred_ssim', 0)
            new_df['ssim_improvement'] = new_df['ssim'] - new_df['blurred_ssim']
        
        new_df['path_original'] = df['original']
        new_df['path_blurred'] = df['blurred']
        new_df['path_restored'] = df.get(algo_name, '')
        
        new_df['test_case_id'] = new_df['image'] + " (" + new_df['filter'] + ")"
        
        return new_df

    raise ValueError("Неизвестный формат CSV")