import pandas as pd
import os

def filename_no_ext(path):
    return os.path.splitext(os.path.basename(str(path)))[0]

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
        
    df = pd.read_csv(csv_path)

    df.columns = [c.strip().replace(' ', '_') for c in df.columns]

    if 'blurred_psnr' not in df.columns: df['blurred_psnr'] = 0
    if 'blurred_ssim' not in df.columns: df['blurred_ssim'] = 0

    if 'algorithm' in df.columns:
        # Считаем прирост PSNR
        if 'psnr_improvement' not in df.columns and 'psnr' in df.columns:
            df['psnr_improvement'] = df['psnr'] - df['blurred_psnr']
            
        # Считаем прирост SSIM
        if 'ssim_improvement' not in df.columns and 'ssim' in df.columns:
            df['ssim_improvement'] = df['ssim'] - df['blurred_ssim']

        df['test_case_id'] = df.apply(
            lambda x: f"{filename_no_ext(x.get('image', ''))} ({filename_no_ext(x.get('filter', ''))})",
            axis=1
        )
        return df

    psnr_cols = [c for c in df.columns if c.startswith('psnr_') 
                 and c not in ['blurred_psnr', 'psnr_improvement']]
    
    if psnr_cols:
        dfs_list = []
        
        for p_col in psnr_cols:
            algo_name = p_col.replace('psnr_', '')
            
            sub_df = pd.DataFrame()
            
            sub_df['image'] = df['original'].apply(filename_no_ext)
            if 'kernel_blur' in df.columns:
                sub_df['filter'] = df['kernel_blur'].apply(filename_no_ext)
            elif 'original_kernel' in df.columns:
                sub_df['filter'] = df['original_kernel'].apply(filename_no_ext)
            else:
                 sub_df['filter'] = 'unknown'

            sub_df['algorithm'] = algo_name
            
            sub_df['psnr'] = df[p_col]
            sub_df['blurred_psnr'] = df['blurred_psnr']
            sub_df['psnr_improvement'] = sub_df['psnr'] - sub_df['blurred_psnr']
            
            ssim_col = f'ssim_{algo_name}'
            if ssim_col in df.columns:
                sub_df['ssim'] = df[ssim_col]
                sub_df['blurred_ssim'] = df.get('blurred_ssim', 0)
                sub_df['ssim_improvement'] = sub_df['ssim'] - sub_df['blurred_ssim']
            else:
                sub_df['ssim'] = 0
                sub_df['ssim_improvement'] = 0
                
            sub_df['path_original'] = df.get('original', '')
            sub_df['path_blurred'] = df.get('blurred', '')
            
            if algo_name in df.columns:
                sub_df['path_restored'] = df[algo_name]
            elif f'restored_{algo_name}' in df.columns:
                sub_df['path_restored'] = df[f'restored_{algo_name}']
            else:
                sub_df['path_restored'] = ''

            sub_df['test_case_id'] = sub_df['image'] + " (" + sub_df['filter'] + ")"
            
            dfs_list.append(sub_df)

        if dfs_list:
            final_df = pd.concat(dfs_list, ignore_index=True)
            return final_df

    raise ValueError("Неизвестный формат CSV: не найдены 'algorithm' или 'psnr_ALGO'")