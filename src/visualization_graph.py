import os
import json
import glob
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class ColorMapper:
    """
    Класс для назначения цветов различным class_name.
    Генерирует стабильные цвета на основе хэша имени класса.
    """
    def __init__(self):
        self.colors = {}
        # Можно также задать базовые цвета для часто встречающихся классов
        self.default_color = (0, 255, 255)  # cyan

    def get_color(self, class_name):
        if class_name not in self.colors:
            # Генерируем цвет на основе хэша строки
            hash_val = hash(class_name)
            # Преобразуем в RGB, используя три байта
            r = (hash_val & 0xFF0000) >> 16
            g = (hash_val & 0x00FF00) >> 8
            b = (hash_val & 0x0000FF)
            # Убеждаемся, что цвет достаточно яркий (если нужно)
            self.colors[class_name] = (r, g, b)
        return self.colors[class_name]

def draw_annotations_on_image(image_path, json_path, output_path=None, show=False, font_size=12, color_mapper=None):
    """
    Рисует bounding box'ы из JSON на изображении и сохраняет/показывает результат.
    """
    if color_mapper is None:
        color_mapper = ColorMapper()

    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    width, height = img.size

    with open(json_path, 'r') as f:
        data = json.load(f)

    nodes = data.get('nodes', [])
    if not nodes:
        # print(f"No nodes in {json_path}")
        return

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for node in nodes:
        data_node = node.get('data', {})
        bbox = data_node.get('bbox_2d', {})
        if not bbox:
            continue
        xyxy = bbox.get('xyxy')
        if not xyxy or len(xyxy) != 4:
            continue

        x1 = xyxy[0] * width
        y1 = xyxy[1] * height
        x2 = xyxy[2] * width
        y2 = xyxy[3] * height

        class_name = data_node.get('class_name', 'unknown')
        color = color_mapper.get_color(class_name)
        # PIL ожидает цвета в виде кортежа (R, G, B) или строки
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 5, y1 + 5), class_name, fill=color, font=font)

    if output_path:
        img.save(output_path)
        print(f"Saved: {output_path}")
    if show:
        plt.imshow(np.array(img))
        plt.axis('off')
        plt.title(os.path.basename(image_path))
        plt.show()

def visualize_all_annotations(image_root, json_root, output_root=None, show=False, verbose=True):
    """
    Обходит все сцены в image_root, находит соответствующие JSON в json_root,
    рисует аннотации и сохраняет/показывает.

    Аргументы:
        image_root (str): корневая папка с изображениями (содержит папку 'scenes').
        json_root (str): корневая папка с JSON-файлами (содержит такую же структуру папок сцен).
        output_root (str): если указан, сохраняет визуализации в эту папку, сохраняя структуру.
        show (bool): показывать ли каждое изображение (осторожно, много окон).
        verbose (bool): печатать прогресс.
    """
    image_scenes_dir = os.path.join(image_root, 'scenes')
    if not os.path.isdir(image_scenes_dir):
        print(f"Папка scenes не найдена: {image_scenes_dir}")
        return

    scene_names = [d for d in os.listdir(image_scenes_dir) if os.path.isdir(os.path.join(image_scenes_dir, d))]
    if verbose:
        print(f"Найдено сцен: {len(scene_names)}")

    color_mapper = ColorMapper()

    for scene in scene_names:
        scene_image_path = os.path.join(image_scenes_dir, scene, 'sequence')
        if not os.path.isdir(scene_image_path):
            continue

        # Путь к JSON для этой сцены
        scene_json_path = os.path.join(json_root, scene)
        if not os.path.isdir(scene_json_path):
            if verbose:
                print(f"JSON папка для сцены {scene} не найдена: {scene_json_path}")
            continue

        # Находим все изображения frame-*.color.jpg
        image_pattern = os.path.join(scene_image_path, "**", "frame-*.color.jpg")
        image_paths = glob.glob(image_pattern, recursive=True)
        image_paths = [p for p in image_paths if '.rendered.' not in p]

        if verbose:
            print(f"Сцена {scene}: {len(image_paths)} изображений")

        for img_path in image_paths:
            base_name = os.path.basename(img_path)
            json_name = base_name.replace('.color.jpg', '.json')
            # Ищем JSON в scene_json_path, возможно рекурсивно, если в подпапках
            json_path = None
            # Поскольку JSON могут быть в подпапках, используем glob для поиска
            json_pattern = os.path.join(scene_json_path, "**", json_name)
            candidates = glob.glob(json_pattern, recursive=True)
            if candidates:
                json_path = candidates[0]
            else:
                if verbose:
                    print(f"JSON не найден: {json_name} в {scene_json_path}")
                continue

            if output_root:
                # Сохраняем в output_root с сохранением структуры относительно image_root
                rel_path = os.path.relpath(img_path, image_root)
                output_path = os.path.join(output_root, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # Изменяем имя, чтобы не перезаписывать исходное
                output_path = output_path.replace('.color.jpg', '.annotated.jpg')
            else:
                output_path = None

            draw_annotations_on_image(img_path, json_path, output_path=output_path, show=show, color_mapper=color_mapper)

# Пример использования
if __name__ == "__main__":
    image_root = "/mnt/external_usb_hdd/6YL/Datasets/3RScan"  # содержит подпапку scenes
    json_root = "/mnt/external_usb_hdd/6YL/Datasets/3RScan/SceneGraphs"           # также содержит подпапку scenes
    output_root = "/home/pinkin_ek/projects/Scene_graph_localization/data/graph_vis"  # если нужно сохранять
    visualize_all_annotations(image_root, json_root, output_root=output_root, show=False, verbose=True)