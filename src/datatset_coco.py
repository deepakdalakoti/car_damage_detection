import glob
import json
import os

import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageDraw


class CarDataset(torch.utils.data.Dataset):
    # Possibly remove small boxes
    def __init__(self, root, mode="train", transforms=None):
        self.root = root

        self.imgs = sorted(glob.glob(os.path.join(root, mode + "/*.jpg")))
        self.mode = mode
        self.transforms = transforms
        if mode in ["train", "val"]:
            self.mode = mode
            self.annot_path = os.path.join(root, mode + "/COCO_" + mode + "_annos.json")
            self.annots = self._read_json(self.annot_path)

    def _read_json(self, path):
        with open(path, "r") as f:
            data = json.loads(f.read())
        return data

    def plot_images(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        img1 = ImageDraw.Draw(img)
        fname = self.imgs[idx].split("/")[-1]
        image_id = [a["id"] for a in self.annots["images"] if a["file_name"] == fname][
            0
        ]
        annots = [
            self.annots["annotations"][i]
            for i in range(len(self.annots["annotations"]))
            if self.annots["annotations"][i]["image_id"] == image_id
        ]
        names = {item["id"]: item["name"] for item in self.annots["categories"]}

        for item in annots:
            item["bbox"][2] = item["bbox"][2] + item["bbox"][0]
            item["bbox"][3] = item["bbox"][3] + item["bbox"][1]

            img1.rectangle(item["bbox"], outline="red")
            img1.text(
                [item["bbox"][0] - 10, item["bbox"][1] - 10], names[item["category_id"]]
            )

        return img

    def __getitem__(self, idx):

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
        # get bounding box coordinates for each mask
        # Target needs class labels and bounding boxes
        boxes = []
        num_objs = len(annots)
        labels = []
        for obj in annots:

            xmin = obj["bbox"][0]
            # xmax = obj["bbox"][0] + obj["bbox"][2]
            xmax = obj["bbox"][2]
            ymin = obj["bbox"][1]
            # ymax = obj["bbox"][1] + obj["bbox"][3]
            ymax = obj["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(obj["category_id"]) - 1)

        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels = torch.as_tensor(labels, dtype=torch.int64)
        area = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        # area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])

        target = {}

        # target["boxes"] = boxes
        # target["class_labels"] = labels
        target["annotations"] = [
            {"bbox": boxes[i], "category_id": labels[i], "area": area[i]}
            for i in range(len(boxes))
        ]

        target["image_id"] = idx

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
