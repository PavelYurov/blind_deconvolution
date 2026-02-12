from typing import List, Dict, Union
import os

def create_manual_visual_block(cases: List[Dict[str, Union[str, float]]]) -> Dict:
    """
    Готовит данные для шаблона visuals.tex из списка словарей, заполненных вручную.
    
    Args:
        cases: Список словарей вида:
        [
            {
                'orig': 'path/to/orig.png',
                'blur': 'path/to/blur.png',
                'res': 'path/to/res.png',
                'psnr': 25.4,
                'ssim': 0.85,
                'algo_name': 'MyAlgo',
                'caption': 'Experiment 1'
            },
            ...
        ]
    """
    items = []
    
    def clean(p): return str(p).replace('\\', '/')
    
    for c in cases:
        items.append({
            'path_original': clean(c.get('orig', '')),
            'path_blurred': clean(c.get('blur', '')),
            'path_restored': clean(c.get('res', '')),
            'filter_name': str(c.get('caption', 'Image')),
            'algorithm': str(c.get('algo_name', 'Method')),
            'psnr': f"{c.get('psnr', 0):.2f}",
            'ssim': f"{c.get('ssim', 0):.3f}",
            'caption_text': str(c.get('caption', 'Visual Comparison'))
        })
        
    return {'items': items}