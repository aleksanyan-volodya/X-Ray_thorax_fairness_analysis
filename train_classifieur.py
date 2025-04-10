from typing import Any
import argparse
from torchvision.models import resnet18, ResNet18_Weights
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.functional import cross_entropy
from torchmetrics.classification import BinaryConfusionMatrix
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    CenterCrop,
    ToImage,
    ToDtype,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
)
from torchvision.datasets import ImageFolder
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, Trainer
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

import io
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--weights_col", type=str, default="WEIGHTS")
    parser.add_argument("--max_epochs", type=int, default=25)
    parser.add_argument("--csv_out", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=False)
    parser.add_argument("--preds_col", type=str, default="preds")
    parser.add_argument("--train", type=str, default="True")
    parser.add_argument("--pred", type=str, default="True")
    return parser.parse_args()

transforms_valid = Compose([
    Resize((256, 256)),
    CenterCrop(224),
    ToImage(),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transforms_train = Compose([
    Resize((256, 256)),
    RandomCrop(224),
    RandomHorizontalFlip(),
    ToImage(),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return balanced_accuracy_score(labels.detach().cpu(), outputs_idx.detach().cpu())

class ChestXRayClassifier(LightningModule):
    def __init__(self, pth_path=None, adamax=False, cosine=True, nb_classes=40) -> None:
        super().__init__()
        self.model = make_model(pth_path, nb_classes)
        self.adamax = adamax
        self.cosine = cosine
        self.bcm = BinaryConfusionMatrix()

    def plot_cm(self, logits, labels, name, batch_idx):
        self.bcm(torch.argmax(logits, dim=1), labels)
        fig, ax = plt.subplots(figsize=(10, 10))
        self.bcm.plot(ax=ax, labels=["malade", "sain"])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        im = Compose([ToImage(), ToDtype(torch.float32, scale=True)])(Image.open(buf))
        self.logger.experiment.add_image(name, im, batch_idx)
        self.bcm.reset()

    def forward(self, imgs):
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.forward(imgs)
        loss = cross_entropy(logits, labels)
        self.log("train_loss", loss)
        acc = accuracy(logits, labels)
        self.log("train_acc", acc)
        self.plot_cm(logits, labels, "train_confusion_matrix", batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.forward(imgs)
        loss = cross_entropy(logits, labels)
        self.log("val_loss", loss)
        acc = accuracy(logits, labels)
        self.log("val_acc", acc)
        self.plot_cm(logits, labels, "valid_confusion_matrix", batch_idx)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adamax(self.parameters(), lr=1e-3, weight_decay=0.001) if self.adamax else torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 1e-7, -1, True)
            if self.cosine else
            torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 9, 15], gamma=0.2)
        )
        return [optimizer], [scheduler]

def make_model(pth_path=None, nb_classes=40, V1=False):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if V1 else ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, nb_classes)
    if pth_path is not None:
        model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    return model

def get_weights(imgs, img_names, img_weights, weight=None):
    sorted_weights = []
    if weight is not None:
        for _, target in imgs:
            sorted_weights.append(weight[target])
    else:
        img_names = img_names.str.strip()
        for img_path, _ in imgs:
            filename = os.path.basename(img_path).strip()
            match = img_names == filename
            if match.any():
                idx = match.idxmax()
                sorted_weights.append(img_weights[idx])
            else:
                print(f"⚠️ Image {filename} non trouvée dans le CSV !")
    return torch.tensor(sorted_weights, dtype=torch.float64)

def preds_todf(df, dataset, label_decoder, model, preds_col):
    for idx in range(len(dataset.imgs)):
        img_name = os.path.basename(dataset.imgs[idx][0]).strip()
        img, label = dataset.__getitem__(idx)
        with torch.no_grad():
            pred_logits = model(img.unsqueeze(0))
            pred = pred_logits.argmax(1).item()
        df.loc[df["Image Index"] == img_name, preds_col] = label_decoder[pred]
        df.loc[df["Image Index"] == img_name, "labels"] = label_decoder[label]
    return df

def pred_classifier(datadir: str, ckpt_path: str, csv_in: str, csv_out: str, preds_col: str = "preds", sensitive_col: str = "Patient Gender"):
    train_dataset = ImageFolder(f"{datadir}/train/", transform=transforms_valid)
    val_dataset = ImageFolder(f"{datadir}/valid/", transform=transforms_valid)
    
    label_encoder = train_dataset.class_to_idx
    label_decoder = {v: k for k, v in label_encoder.items()}

    model = ChestXRayClassifier(adamax=True, cosine=True, nb_classes=len(label_encoder))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    model.eval()

    df = pd.read_csv(csv_in)
    df["Image Index"] = df["Image Index"].apply(lambda x: os.path.basename(x))
    df[preds_col] = None

    print("Start prediction on train dataset")
    df = preds_todf(df, train_dataset, label_decoder, model, preds_col)

    print("Start prediction on validation dataset")
    df = preds_todf(df, val_dataset, label_decoder, model, preds_col)

    df.to_csv(csv_out, index=False)

    if "labels" in df.columns and df[preds_col].notna().all():
        print("\n📊 Global Metrics:")
        print("- Balanced Accuracy:", round(balanced_accuracy_score(df.labels, df[preds_col]), 4))
        print("- Accuracy:", round(accuracy_score(df.labels, df[preds_col]), 4))

        if sensitive_col in df.columns:
            print(f"\n📊 Fairness Metrics (by {sensitive_col}):")

            for group in df[sensitive_col].dropna().unique():
                group_df = df[df[sensitive_col] == group]
                acc = accuracy_score(group_df["labels"], group_df[preds_col])
                bal_acc = balanced_accuracy_score(group_df["labels"], group_df[preds_col])
                cm = confusion_matrix(group_df["labels"], group_df[preds_col], labels=list(label_encoder.keys()))
                tp = cm[1][1] if cm.shape == (2, 2) else 0
                fn = cm[1][0] if cm.shape == (2, 2) else 0
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

                print(f"  ▶ Group '{group}':")
                print(f"     - Accuracy: {round(acc, 4)}")
                print(f"     - Balanced Accuracy: {round(bal_acc, 4)}")
                print(f"     - TPR (Sensitivity): {round(tpr, 4)}")

            # Disparity between groups
            groups = df[sensitive_col].dropna().unique()
            if len(groups) == 2:
                g1, g2 = groups
                acc1 = accuracy_score(df[df[sensitive_col] == g1]["labels"], df[df[sensitive_col] == g1][preds_col])
                acc2 = accuracy_score(df[df[sensitive_col] == g2]["labels"], df[df[sensitive_col] == g2][preds_col])
                print(f"\n⚖️  Accuracy Disparity ({g1} vs {g2}): {round(abs(acc1 - acc2), 4)}")
    else:
        print("⚠️ Impossible de calculer les métriques (labels ou prédictions manquants)")

def train_classifier(logdir: str, datadir: str, csv: str, weights_col: str = "WEIGHTS", max_epochs: int = 25):
    t = time()
    train_datadir = f"{datadir}/train/"
    valid_datadir = f"{datadir}/valid/"
    df = pd.read_csv(csv)
    logger = TensorBoardLogger(f"{logdir}/", name="classifier")
    cb_ckpt_best = ModelCheckpoint(dirpath=f"{logdir}/", monitor="val_loss", filename="best-val-loss", mode="min", save_top_k=1, save_last=False)
    cb_es = EarlyStopping(monitor="val_loss", patience=3)
    trainer = Trainer(logger=logger, callbacks=[cb_ckpt_best, cb_es], max_epochs=max_epochs)
    train_dataset = ImageFolder(train_datadir, transform=transforms_train)
    val_dataset = ImageFolder(valid_datadir, transform=transforms_valid)
    train_weights = get_weights(train_dataset.imgs, df["Image Index"], df[weights_col])
    valid_weights = get_weights(val_dataset.imgs, df["Image Index"], df[weights_col])
    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=WeightedRandomSampler(train_weights, len(train_weights)))
    val_dataloader = DataLoader(val_dataset, batch_size=32, sampler=WeightedRandomSampler(valid_weights, len(valid_weights)))
    model = ChestXRayClassifier(adamax=True, cosine=True, nb_classes=len(train_dataset.class_to_idx))
    print("Start training")
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print(f"End of training {time()-t}")
    return cb_ckpt_best.best_model_path, cb_ckpt_best.best_model_score.item()

if __name__ == "__main__":
    args = parse_opt()
    if args.train == "True":
        ckpt_path, ckpt_score = train_classifier(
            logdir=args.logdir,
            datadir=args.datadir,
            csv=args.csv,
            weights_col=args.weights_col,
            max_epochs=args.max_epochs,
        )
        print(ckpt_path, ckpt_score)
        if args.pred == "True":
            pred_classifier(
                datadir=args.datadir,
                ckpt_path=ckpt_path,
                csv_in=args.csv,
                csv_out=args.csv_out,
                preds_col=args.preds_col,
            )
    elif args.pred == "True":
        pred_classifier(
            datadir=args.datadir,
            ckpt_path=args.ckpt_path,
            csv_in=args.csv,
            csv_out=args.csv_out,
            preds_col=args.preds_col,
        )
