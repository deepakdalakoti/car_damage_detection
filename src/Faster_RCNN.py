import numpy as np
import torch
import torchvision
from packaging import version
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import dependencies.transforms as T
import dependencies.utils as utils
from dependencies.engine import evaluate, train_one_epoch
from src.dataset import CarDataset

if version.parse(np.__version__) >= version.parse("1.24.0"):
    np.float = np.float32


writer = SummaryWriter("./logs")


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_instance_object_model(num_classes):
    # load pretrained faster-rcnn model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train(data_dir):

    dataset = CarDataset(data_dir, "train", get_transform(train=True))
    dataset_val = CarDataset(data_dir, "val", get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=3, shuffle=True, num_workers=2, collate_fn=utils.collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        collate_fn=utils.collate_fn,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 6

    # get the model using our helper function
    model = get_instance_object_model(num_classes)
    print(device)
    # move model to the right device
    model.to(device)

    images, labels = next(iter(data_loader))
    # writer.add_graph(model, (images, labels))

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 20

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metrics = train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10
        )
        writer.add_scalar("training_loss", metrics.loss.avg, epoch)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        metrics_eval, _ = evaluate(model, data_loader_val, device=device)
        # print(metrics_eval.coco_eval['bbox'].stats)
        writer.add_scalar(
            "Average precision (eval)", metrics_eval.coco_eval["bbox"].stats[0], epoch
        )

    save_model(model, "car_damage_model.pt")


def save_model(model, fname):
    torch.save(model, fname)
    return


def load_model(path):
    return torch.load(path)


def predict(model, img, device="cpu"):
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    return prediction


def plot_predictions(img, prediction, id_mapping, threshold=0.6):

    img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    img2 = ImageDraw.Draw(img1)
    cats = {item["id"]: item["name"] for item in id_mapping}
    # print(cats)
    for box, score, labels in zip(
        prediction[0]["boxes"], prediction[0]["scores"], prediction[0]["labels"]
    ):
        if score < threshold:
            continue
        imgbox = box.cpu().numpy()
        img2.rectangle(imgbox, outline="red")
        labs = int(labels.cpu())
        text = cats[labs] + f" {score:.2f}"
        img2.text([imgbox[0] - 10, imgbox[1] - 10], text, fill="blue")
    return img1
