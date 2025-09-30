# analyze_results.py (Versione che scansiona le sottocartelle)

import os
import json
import pandas as pd

# --- Configuration ---
RESULTS_DIR = 'results'
REPORT_FILENAME = 'report.csv'

def analyze():
    """
    Scansiona ricorsivamente tutte le sottocartelle in 'results',
    carica i file '_metadata.json', li aggrega e salva un report.
    """
    all_results = []
    print(f"Scansione ricorsiva della cartella '{RESULTS_DIR}' in corso...")

    for root, dirs, files in os.walk(RESULTS_DIR):
        for filename in files:
            if filename.endswith('_metadata.json'):
                filepath = os.path.join(root, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        
                        flat_data = {
                            'image': data.get('parameters', {}).get('image'),
                            'method': data.get('method'),
                            'kernel_size': data.get('parameters', {}).get('kernel_size'),
                            'lambda_prior': data.get('parameters', {}).get('lambda_prior'),
                            'lambda_kernel': data.get('parameters', {}).get('lambda_kernel'),
                            'iterations': data.get('parameters', {}).get('iterations'),
                            'blur_type': data.get('parameters', {}).get('blur_type', 'N/A'),
                            'motion_len': data.get('parameters', {}).get('motion_len', 'N/A'),
                            'gaussian_sigma': data.get('parameters', {}).get('gaussian_sigma', 'N/A'),
                            'exec_time_sec': data.get('execution_time_sec'),
                            'input_sharpness': data.get('input_sharpness'),
                            'output_sharpness': data.get('output_sharpness'),
                            'psnr': data.get('metrics', {}).get('psnr'),
                            'ssim': data.get('metrics', {}).get('ssim'),
                            'result_folder': os.path.basename(root) # Aggiungiamo il nome della cartella per un facile ritrovamento!
                        }
                        all_results.append(flat_data)
                except Exception as e:
                    print(f"Errore durante l'elaborazione del file {filepath}: {e}")

    if not all_results:
        print(f"Nessun file '_metadata.json' trovato nelle sottocartelle di '{RESULTS_DIR}'.")
        return

    df = pd.DataFrame(all_results)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 160)
    pd.set_option('display.precision', 4)

    print("\n--- Summary of All Executed Tests ---")
    
    columns_to_show = [
        'image', 'kernel_size', 'lambda_prior', 
        'blur_type', 'psnr', 'ssim', 'output_sharpness', 'result_folder'
    ]
    display_columns = [col for col in columns_to_show if col in df.columns]
    print(df[display_columns])
    
    try:
        df.sort_values(by=['image', 'psnr'], ascending=[True, False]).to_csv(REPORT_FILENAME, index=False)
        print(f"\nReport completo salvato con successo in '{REPORT_FILENAME}'")
    except Exception as e:
        print(f"\nErrore durante il salvataggio del report CSV: {e}")


if __name__ == "__main__":
    print("Executing results analysis script...")
    try:
        import pandas
    except ImportError:
        print("\nWARNING: The 'pandas' library is not installed.")
        print("Please run this from your terminal: pip install pandas")
        exit()
        
    analyze()