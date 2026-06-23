import os
import io
import tokenize

# =====================================================================
# НАСТРОЙКИ
# =====================================================================
# Укажи список всех папок, которые нужно очистить
TARGET_DIRECTORIES = [
    # # # r"adaptive_euler_elastica/aeer",
    # # # r"adaptive_euler_elastica/aeer_poisson",
    # # # # r"amape_htp",
    # # r"bid_hbsp_exp/bid_hbsp_babacan",
    # # r"bid_hbsp_exp/bid_hbsp_denoise_experiment",
    # # r"bid_hbsp_exp/bid_hbsp_babacan/bid_hbsp_bcsnsp_sr",
    # # r"gbbid_exp/gbbid_denoise_experiment",
    # # r"lip_exp/lip_denoise_experiment",
    # # # # r"mrf",
    # # # # r"nscp",
    # # # r"fractional_order",
    # # r"pmp_exp/pmp_denoise_experiment",
    # # r"pmp_exp/pmp_denoise_fix",
    # # r"pmp_exp/pmp_denoise_improved",
    # # r"pmp_exp/pmp_denoise_merge",
    # r"amape_htp",
    # r"bayesian/ard",
    # r"bayesian/cgmrf",
    # r"bayesian/eml",
    # r"bayesian/hsp",
    # r"bayesian/rcs",
    # r"bayesian/vbskb_bid_sp",
    # r"bayesian/vdbke/vdbke",
    r"bayesian/vdbke/vdbke_cython",
    r"bayesian/vdbke/vdbke_denoise",
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

]

REMOVE_COMMENTS = True    # Удалять все `# комментарии`
REMOVE_DOCSTRINGS = True  # Удалять """докстринги""" внутри кода
KEEP_MODULE_DOC = True    # Оставить самый первый """докстринг""" в файле

# Расширения файлов для обработки
VALID_EXTENSIONS = {".py", ".pyx"}
# =====================================================================

def clean_python_code(source_code):
    """
    Разбирает код на токены, безопасно удаляет комментарии и докстринги,
    собирает код обратно и форматирует пустые строки.
    """
    source_bytes = source_code.encode('utf-8')
    try:
        tokens = list(tokenize.tokenize(io.BytesIO(source_bytes).readline))
    except Exception as e:
        print(f"  [Ошибка токенизатора, пропускаю]: {e}")
        return source_code

    out_tokens = []
    is_first_statement = True

    for i, tok in enumerate(tokens):
        # 1. Удаляем обычные комментарии (# ...)
        if REMOVE_COMMENTS and tok.type == tokenize.COMMENT:
            continue

        # 2. Обработка многострочных строк (докстрингов)
        if tok.type == tokenize.STRING:
            # Пытаемся понять, является ли строка "самостоятельной" (т.е. докстрингом)
            # или это значение переменной (например, a = "текст")
            prev_meaningful = None
            for j in range(i - 1, -1, -1):
                if tokens[j].type not in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING):
                    prev_meaningful = tokens[j]
                    break

            next_meaningful = None
            for j in range(i + 1, len(tokens)):
                if tokens[j].type not in (tokenize.COMMENT, tokenize.NL):
                    next_meaningful = tokens[j]
                    break

            is_standalone = False
            if prev_meaningful is None or prev_meaningful.type in (tokenize.INDENT, tokenize.NEWLINE):
                if next_meaningful is None or next_meaningful.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
                    is_standalone = True

            if is_standalone:
                if KEEP_MODULE_DOC and is_first_statement:
                    # Это самый первый докстринг файла — оставляем его
                    is_first_statement = False
                    out_tokens.append(tok)
                    continue
                elif REMOVE_DOCSTRINGS:
                    # Это докстринг функции/класса — удаляем
                    continue

        # Как только мы встретили первый реальный код,
        # мы больше не на самом верхнем уровне модуля.
        if tok.type not in (tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
            is_first_statement = False

        out_tokens.append(tok)

    # Собираем токены обратно в строку
    try:
        new_bytes = tokenize.untokenize(out_tokens)
        new_code = new_bytes.decode('utf-8')
    except Exception as e:
        print(f"  [Ошибка сборки кода, пропускаю]: {e}")
        return source_code

    # 3. Косметическая очистка: удаляем лишние пустые строки
    lines = new_code.splitlines()
    cleaned = []
    for line in lines:
        line = line.rstrip() # убираем пробелы на концах
        
        # Ограничиваем количество подряд идущих пустых строк до 2
        if not line:
            if len(cleaned) >= 2 and not cleaned[-1] and not cleaned[-2]:
                continue
        cleaned.append(line)

    # Убираем пустые строки в самом начале и в самом конце
    while cleaned and not cleaned[0]:
        cleaned.pop(0)
    while cleaned and not cleaned[-1]:
        cleaned.pop()

    return "\n".join(cleaned) + "\n"


def main():
    total_processed = 0

    for target_dir in TARGET_DIRECTORIES:
        # Убираем лишние слеши для красоты вывода
        target_dir = target_dir.rstrip(os.sep)
        
        if not os.path.exists(target_dir):
            print(f"\n! ПРЕДУПРЕЖДЕНИЕ: Папка не найдена, пропускаю: {target_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Обработка папки: {target_dir}")
        print(f"{'='*60}")
        
        folder_processed = 0
        
        for root, _, files in os.walk(target_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in VALID_EXTENSIONS:
                    filepath = os.path.join(root, file)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        original_code = f.read()

                    cleaned_code = clean_python_code(original_code)

                    if cleaned_code != original_code:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(cleaned_code)
                        print(f"  [Очищен] {file}")
                        folder_processed += 1
                        total_processed += 1
        
        if folder_processed == 0:
            print("  В этой папке нечего очищать или файлы уже чистые.")

    print(f"\nV Готово! Всего очищено файлов во всех папках: {total_processed}")

if __name__ == "__main__":
    main()