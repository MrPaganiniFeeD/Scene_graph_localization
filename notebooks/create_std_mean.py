import os
from glob import glob
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm  # для красивого прогресс-бара (можно установить pip install tqdm)

def compute_dataset_mean_std(dataset_folder, resize_to=None, verbose=True):
    """
    Вычисляет среднее и стандартное отклонение для всех изображений в датасете.
    
    Аргументы:
        dataset_folder (str): путь к корневой папке датасета, внутри которой находится папка 'scenes'.
        resize_to (tuple or None): если указан (например, (224, 224)), изображения будут приведены к этому размеру
                                    перед вычислением статистик.
        verbose (bool): выводить ли прогресс и промежуточную информацию.
    
    Возвращает:
        mean (list[float]): среднее для каналов RGB.
        std (list[float]): стандартное отклонение для каналов RGB.
    """
    scenes_dir = os.path.join(dataset_folder, 'scenes')
    if not os.path.isdir(scenes_dir):
        raise FileNotFoundError(f"Папка со сценами не найдена: {scenes_dir}")

    # Получаем список всех сцен (подпапок внутри scenes)
    scene_names = [d for d in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, d))]
    if verbose:
        print(f"Найдено сцен: {len(scene_names)}")

    # Если нужен ресайз, создадим трансформ
    transform_list = []
    if resize_to is not None:
        transform_list.append(transforms.Resize(resize_to))
    transform_list.append(transforms.ToTensor())  # конвертирует в тензор (C, H, W) со значениями [0, 1]
    transform = transforms.Compose(transform_list)

    # Инициализируем аккумуляторы
    total_pixels = 0
    sum_r = sum_g = sum_b = 0.0
    sum_sq_r = sum_sq_g = sum_sq_b = 0.0

    # Пройдём по всем сценам
    for scene in tqdm(scene_names, desc="Обработка сцен", disable=not verbose):
        scene_path = os.path.join(scenes_dir, scene, 'sequence')
        if not os.path.isdir(scene_path):
            continue

        # Ищем все файлы frame-*.color.jpg внутри sequence (включая подпапки)
        pattern = os.path.join(scene_path, "**", "frame-*.color.jpg")
        image_paths = glob(pattern, recursive=True)
        # Отфильтровываем .rendered., если нужно (как в вашем коде)
        image_paths = [p for p in image_paths if '.rendered.' not in p]

        if verbose and len(image_paths) == 0:
            print(f"Внимание: в сцене {scene} не найдено изображений.")

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_rotated_cw = img.transpose(Image.ROTATE_270)
            except Exception as e:
                if verbose:
                    print(f"Ошибка загрузки {img_path}: {e}")
                continue

            # Применяем трансформации (ресайз + to_tensor)
            tensor = transform(img_rotated_cw)  # shape: (3, H, W), значения в [0,1]
            # Приводим к виду (C, N) где N = H*W
            pixels = tensor.view(3, -1)

            # Суммируем пиксели и квадраты пикселей
            sum_r += pixels[0].sum().item()
            sum_g += pixels[1].sum().item()
            sum_b += pixels[2].sum().item()

            sum_sq_r += (pixels[0] ** 2).sum().item()
            sum_sq_g += (pixels[1] ** 2).sum().item()
            sum_sq_b += (pixels[2] ** 2).sum().item()

            total_pixels += pixels.size(1)

    # Вычисляем средние и стандартные отклонения
    mean_r = sum_r / total_pixels
    mean_g = sum_g / total_pixels
    mean_b = sum_b / total_pixels

    std_r = ((sum_sq_r / total_pixels) - (mean_r ** 2)) ** 0.5
    std_g = ((sum_sq_g / total_pixels) - (mean_g ** 2)) ** 0.5
    std_b = ((sum_sq_b / total_pixels) - (mean_b ** 2)) ** 0.5

    mean = [mean_r, mean_g, mean_b]
    std = [std_r, std_g, std_b]

    if verbose:
        print(f"\nВычисленные статистики (всего пикселей: {total_pixels}):")
        print(f"mean = {mean}")
        print(f"std  = {std}")

    return mean, std

# Пример использования:
if __name__ == "__main__":
    dataset_folder = "/mnt/external_usb_hdd/6YL/Datasets/3RScan"  # замените на реальный путь
    # Если хотите ресайзить перед подсчётом, передайте resize_to=(224,224)
    mean, std = compute_dataset_mean_std(dataset_folder, resize_to=(322,322))
    print(mean, std)