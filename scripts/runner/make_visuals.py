import argparse
import sys
import os
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists() and not (path / "src").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root (pyproject.toml or src/ not found)")
        path = path.parent
    return path

current_file = Path(__file__).resolve()
try:
    project_root = _find_project_root(current_file)
except RuntimeError as e:
    print(f"Error: {e}")
    sys.exit(1)

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.blinddeconv.output.exporter import LatexExporter
from src.blinddeconv.output.visuals import create_manual_visual_block


def main():
    parser = argparse.ArgumentParser(
        description="""
        Генератор LaTeX блока визуального сравнения для отчета.
        Создает .tex файл с тремя изображениями в ряд: Original | Blurred | Restored.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', '--input_image', '--orig', required=True, dest='orig',
                        help='Путь к ОРИГИНАЛЬНОМУ изображению')
    
    parser.add_argument('-b', '--blurred_image', '--blur', required=True, dest='blur',
                        help='Путь к СМАЗАННОМУ изображению')
    
    parser.add_argument('-r', '--restored_image', '--rest', required=True, dest='rest',
                        help='Путь к ВОССТАНОВЛЕННОМУ изображению')

    parser.add_argument('-p', '--psnr', type=float, default=0.0,
                        help='Значение PSNR для подписи (по умолчанию: 0.0)')
    
    parser.add_argument('-s', '--ssim', type=float, default=0.0,
                        help='Значение SSIM для подписи (по умолчанию: 0.0)')

    parser.add_argument('-a', '--algorithm', type=str, default="Algorithm", dest='algo',
                        help='Название алгоритма (отображается под 3-й картинкой)')
    
    parser.add_argument('-f', '--filter_name', type=str, default="Blur", dest='filter_name',
                        help='Название типа смаза (отображается под 2-й картинкой)')
    
    parser.add_argument('-c', '--caption', type=str, default="",
                        help='Общая подпись к рисунку. Если пусто, берется из названия оригинала.')

    # Настройки сохранения
    parser.add_argument('-o', '--out_filename', type=str, default="visual_block",
                        help='Имя выходного файла без расширения (по умолчанию: visual_block)')
    
    parser.add_argument('-d', '--out_dir', type=str, default="reports/manual_blocks",
                        help='Папка для сохранения (по умолчанию: reports/manual_blocks)')

    args = parser.parse_args()

    full_output_dir = os.path.join(project_root, args.out_dir)
    exporter = LatexExporter(output_dir=full_output_dir)

    print(f"--- Генерация ---")
    print(f"Алгоритм: {args.algo}")
    print(f"PSNR: {args.psnr}, SSIM: {args.ssim}")

    case_data = [{
        'orig': args.orig,
        'blur': args.blur,
        'res': args.rest,
        'psnr': args.psnr,
        'ssim': args.ssim,
        'algo_name': args.algo,
        'caption': args.filter_name 
    }]
    
    context = create_manual_visual_block(case_data)
    
    if args.caption:
        if context['items']:
            context['items'][0]['caption_text'] = args.caption
    elif not args.caption:
        orig_name = os.path.splitext(os.path.basename(args.orig))[0]
        context['items'][0]['caption_text'] = f"Restoration results for {orig_name}"

    try:
        exporter.save_visuals(context, args.out_filename)
        
        output_file = os.path.join(full_output_dir, f"{args.out_filename}.tex")
        print(f"\n[УСПЕШНО] Файл создан: {output_file}")
        print(f"Для вставки в отчет используйте:")
        print(f"\\input{{{args.out_dir}/{args.out_filename}}}")
        
    except Exception as e:
        print(f"\n[ОШИБКА] Не удалось создать файл: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()