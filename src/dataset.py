import glob
import json
import os

import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageDraw


class CarDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transforms=None):
        self.root = root
        self.transforms = transforms
        # Store list of image names
        self.imgs = sorted(glob.glob(os.path.join(root, mode + "/*.jpg")))
        self.mode = mode
        # For other modes there will ne no annotations
        if mode in ["train", "val"]:
            self.mode = mode
            self.annot_path = os.path.join(
                root, mode + "/COCO_mul_" + mode + "_annos.json"
            )
            self.annots = self._read_json(self.annot_path)

    def _read_json(self, path):
        with open(path, "r") as f:
            data = json.loads(f.read())
        return data

    def plot_images(self, idx):
        img, annots = self[idx]
        img1 = ImageDraw.Draw(img)
        names = {item["id"]: item["name"] for item in self.annots["categories"]}

        for box, label in zip(annots["boxes"], annots["labels"]):
            img1.rectangle(box.numpy(), outline="red")
            img1.text([box[0] - 10, box[1] - 10], names[int(label)])

        return img

    def __getitem__(self, idx):
        # load images
        img = Image.open(self.imgs[idx]).convert("RGB")
        if self.mode not in ["train", "val"]:
            target = {}
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target

        fname = self.imgs[idx].split("/")[-1]
        image_id = [a["id"] for a in self.annots["images"] if a["file_name"] == fname][
            0
        ]
        annots = [
            self.annots["annotations"][i]
            for i in range(len(self.annots["annotations"]))
            if self.annots["annotations"][i]["image_id"] == image_id
        ]
        obj_ids = [a["id"] for a in self.annots["categories"]]
        # get bounding box coordinates
        boxes = []
        num_objs = len(annots)
        labels = []
        for obj in annots:
            # Boxes are provided as x0, y0, w, h -> convert to coordinates as needed by Faster-RCNN
            xmin = obj["bbox"][0]
            xmax = obj["bbox"][0] + obj["bbox"][2]
            ymin = obj["bbox"][1]
            ymax = obj["bbox"][1] + obj["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(obj["category_id"]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([image_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
