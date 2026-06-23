import os

# =====================================================================
# НАСТРОЙКИ
# =====================================================================
TARGET_DIRECTORIES = [
    # r"adaptive_euler_elastica/aeer",
    # r"adaptive_euler_elastica/aeer_poisson",
    # r"amape_htp",
    # r"bid_hbsp_exp/bid_hbsp_babacan",
    # r"bid_hbsp_exp/bid_hbsp_denoise_experiment",
    # r"bid_hbsp_exp/bid_hbsp_babacan/bid_hbsp_bcsnsp_sr",
    # r"gbbid_exp/gbbid_denoise_experiment",
    # r"lip_exp/lip_denoise_experiment",
    # r"mrf",
    # r"nscp",
    # r"fractional_order",
    # r"pmp_exp/pmp_denoise_experiment",
    # r"pmp_exp/pmp_denoise_fix",
    # r"pmp_exp/pmp_denoise_improved",
    # r"pmp_exp/pmp_denoise_merge",
    # r"amape_htp",
    # r"bayesian/ard",
    # r"bayesian/cgmrf",
    # r"bayesian/eml",
    # r"bayesian/hsp",
    # r"bayesian/rcs",
    # r"bayesian/vbskb_bid_sp",
    # r"bayesian/vdbke/vdbke",
    # r"bayesian/vdbke/vdbke_cython",
    # r"bayesian/vdbke/vdbke_denoise",
    # r"bdgsp",
    # r"fractional_order",
    # r"low_rank_hyper_laplacian/lowrank",
    # r"low_rank_hyper_laplacian/lowrank_denoise",
    # r"low_rank_hyper_laplacian/lowrank_li",
    # r"mhdm",
    # r"mrf",
    # r"nscp",
    # r"nsm",
    # r"oid",
    # r"pam/pam",
    # r"pam/pam_denoise",
    # r"pam/pam_cython",
    # r"prida",
    r"_sanitation_cascade"
]

VALID_EXTENSIONS = {".py", ".pyx"}
# =====================================================================

def squeeze_empty_lines(source_code):
    lines = source_code.splitlines()
    clean_lines = []
    
    for line in lines:
        line_to_add = line.rstrip() # Убираем пробелы справа (превращает строку из пробелов в полностью пустую)
        
        if not line_to_add:
            # Если текущая строка пустая, и предыдущая ТОЖЕ пустая — пропускаем
            if clean_lines and not clean_lines[-1]:
                continue
            clean_lines.append("")
        else:
            clean_lines.append(line_to_add)

    # Дополнительно убираем пустые строки в самом начале и конце файла
    while clean_lines and not clean_lines[0]:
        clean_lines.pop(0)
    while clean_lines and not clean_lines[-1]:
        clean_lines.pop()

    return "\n".join(clean_lines) + "\n"

def main():
    total_fixed = 0

    for target_dir in TARGET_DIRECTORIES:
        target_dir = target_dir.rstrip(os.sep)
        if not os.path.exists(target_dir):
            continue
            
        print(f"Сжимаем строки в: {target_dir}")
        folder_fixed = 0
        
        for root, _, files in os.walk(target_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in VALID_EXTENSIONS:
                    filepath = os.path.join(root, file)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        original_code = f.read()

                    fixed_code = squeeze_empty_lines(original_code)

                    if fixed_code != original_code:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(fixed_code)
                        print(f"  [Уплотнён] {file}")
                        folder_fixed += 1
                        total_fixed += 1
                        
        if folder_fixed == 0:
            print("  Всё и так плотно.")

    print(f"\nV Готово! Код сжат в {total_fixed} файлах.")

if __name__ == "__main__":
    main()