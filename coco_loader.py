import json
import os
import cv2


class COCODataset:
    def __init__(self, json_path, img_dir):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.img_dir = img_dir
        self.images = {img['id']: img for img in self.data.get('images', [])}

        self.person_cat_id = None
        for cat in self.data.get('categories', []):
            if cat.get('name') == 'person':
                self.person_cat_id = cat.get('id')
                break
        if self.person_cat_id is None:
            raise ValueError("COCO annotations do not contain a 'person' category")

        self.annotations = {}
        for ann in self.data.get('annotations', []):
            if ann.get('category_id') == self.person_cat_id:
                img_id = ann.get('image_id')
                self.annotations.setdefault(img_id, []).append(ann)

    def get_image(self, img_id):
        img_info = self.images[img_id]
        path = os.path.join(self.img_dir, img_info['file_name'])
        img = cv2.imread(path)
        return img, img_info

    def get_boxes(self, img_id):
        anns = self.annotations.get(img_id, [])
        boxes = [a['bbox'] for a in anns]
        return boxes

    def visualize(self, img_id, save_path=None):
        img, info = self.get_image(img_id)
        if img is None:
            raise FileNotFoundError(f"Image not found: {info['file_name']}")

        boxes = self.get_boxes(img_id)
        for (x, y, w, h) in boxes:
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img)

        return img

    def get_image_ids(self):
        return list(self.images.keys())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize COCO person annotations')
    parser.add_argument('--json', default=os.path.join('data', 'annotations.json'), help='Path to COCO annotations JSON')
    parser.add_argument('--images', default=os.path.join('data', 'images'), help='Image directory')
    parser.add_argument('--output', default=os.path.join('outputs', 'ground_truth'), help='Output directory')
    args = parser.parse_args()

    dataset = COCODataset(args.json, args.images)
    os.makedirs(args.output, exist_ok=True)

    for img_id in dataset.get_image_ids():
        out_path = os.path.join(args.output, f"{img_id}.jpg")
        dataset.visualize(img_id, save_path=out_path)
