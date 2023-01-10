import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.datatset_coco import CarDataset

#TODO: Convert to pytorch lightning
#TODO: train on a bigger dataset
#TODO: Get proper evaluation metrics which are consistent across yolos and FasterRCNN
#TODO: Save huggingface way
#TODO: Add tqdm
#TODO: optimise code

writer = SummaryWriter("./logs")

def collate_with_device(device, feature_extractor):
    def collate_fn(batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        # preprocess image and labels
        processed = feature_extractor(images, annotations=labels, return_tensors="pt")
        processed["labels"] = [
            {k: v.to(device) for k, v in item.items()} for item in processed["labels"]
        ]
        return processed["pixel_values"].to(device), processed["labels"]

    return collate_fn


def train(data_dir):

    feature_extractor = YolosFeatureExtractor.from_pretrained("hustvl/yolos-small")
    model = YolosForObjectDetection.from_pretrained(
        "hustvl/yolos-small", num_labels=1, ignore_mismatched_sizes=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device="cpu"
    collate_fn = collate_with_device(device, feature_extractor)
    dataset = CarDataset(data_dir, "train")
    train_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5, weight_decay=1e-4)
    model.to(device)

    # Training
    num_epochs = 30
    bars = tqdm(range(num_epochs))
    for epochs in range(num_epochs):
        epoch_loss = 0
        epoch_count = 0
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images, labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += outputs.loss_dict['loss_ce'].detach().numpy()
            epoch_count +=1
        #TODO: add eval loss
        print(f"EPOCH {epochs}")
        writer.add_scalar("training_loss", epoch_loss/epoch_count, epochs)
        evaluate(data_dir, model, feature_extractor, device, epochs)
        bars.update(1)
    torch.save(model,"car_damage_model_yolos.pt")


def evaluate(data_dir, model, feature_extractor, device, epoch):
    collate_fn = collate_with_device(device, feature_extractor)
    dataset = CarDataset(data_dir, "val")
    val_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    model.eval()
    loss=0
    with torch.no_grad():
        
        for images, labels in val_loader:
            outputs = model(images, labels)
            #Outputs is not normalised
            loss = loss+ outputs.loss
    writer.add_scalar("validation_loss", loss, epoch)
    return


   

def plot_output(outputs, image, feature_extractor, threshold=0.5):
    # keep only predictions of queries with 0.9+ confidence (excluding no-object class)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    print(probas.max())
    # rescale bounding boxes
    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]["boxes"]
    plot_results(image, probas[keep], bboxes_scaled[keep])


def plot_results(pil_img, prob, boxes):
    COLORS = [
        [0.000, 0.447, 0.741],
        [0.850, 0.325, 0.098],
        [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556],
        [0.466, 0.674, 0.188],
        [0.301, 0.745, 0.933],
    ]
    #id2label = model.config.id2label
    id2label = {0:'damage', 1:'NA'}

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3
            )
        )
        cl = p.argmax()
        text = f"{id2label[cl.item()]}: {p[cl]:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()
