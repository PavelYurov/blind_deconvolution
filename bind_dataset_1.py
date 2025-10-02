import  processing as pr
import os


def bind_dataset(framework: pr.Processing, color = None):
    '''
    сшивает датасет воедино
    '''
    path = 'images\\dataset'
    image_files = [f for f in os.listdir(f'{path}\\original') if os.path.isfile(os.path.join(f'{path}\\original', f))]
    kernel_files = [f for f in os.listdir(f'{path}\\kernel') if os.path.isfile(os.path.join(f'{path}\\kernel', f))]

    for img in image_files:
        for ker in kernel_files:
            # print(img, ker)
            im,_ = os.path.splitext(os.path.basename(img))
            kernel,_ = os.path.splitext(os.path.basename(ker))
            blur = os.path.join(f'{path}\\blurred',f'{im}_{kernel}_img.png')
            framework.bind(os.path.join(f'{path}\\original',img),blur,os.path.join(f'{path}\\kernel',ker),filter_description=f"{kernel}",color=color)

    return framework


