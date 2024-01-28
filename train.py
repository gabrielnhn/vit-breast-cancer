from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
import argparse
import PIL

# MODEL ARGUMENTS
parser = argparse.ArgumentParser(description='Breast cancer classification.')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--batch_size', default=8, type=int)
args = parser.parse_args()

epochs, lr = args.epochs, args.lr
batch_size = args.batch_size

dataset = load_dataset('./breakhis.hf')
metric = evaluate.load("accuracy")

# im = dataset["train"][0]["image"]
# print(dataset["train"][0]["label"])
# # im = PIL.Image(dataset["train"][0]["image"])
# # im.show()
# import matplotlib.pyplot as plt
# plt.imshow(im)
# plt.show()

vitprocessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
#define callback to data collator
def transform(data):
    print(data)
    batch = vitprocessor([PIL.Image.open(x).convert("RGB") for x in data["image_paths"]], return_tensors="pt")
    batch["labels"] = data["label"]
    return batch

def data_collator(unprocessed_batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in unprocessed_batch]),
        "labels": torch.tensor([x["labels"] for x in unprocessed_batch])
    } 

processed_dataset = dataset.with_transform(transform)
training_set = processed_dataset["train"]
test_set = processed_dataset["test"]
training_dataloader = DataLoader(training_set, collate_fn=data_collator, batch_size=batch_size)

test_dataloader = DataLoader(test_set, collate_fn=data_collator, batch_size=batch_size)

## training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', ignore_mismatched_sizes=True, num_labels=2)
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

model.to(device)

for epoch in range(epochs):
    print(f"epoch: {epoch}")
    model.train()
    for _, batch in enumerate(training_dataloader):
        optimizer.zero_grad()

        # print(batch["labels"])
        batch = {k:v.to(device) for k, v in batch.items()} # bring to GPU
        
        outputs = model(**batch)
        loss = outputs.loss 
        loss.backward()
        optimizer.step()


    model.eval()
    for _, batch in enumerate(test_dataloader):
        with torch.no_grad():
            # print(batch["labels"])
            batch = {k:v.to(device) for k, v in batch.items()} # bring to GPU
            
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=preds, references=batch["labels"])


    print(f"accuracy: {metric.compute()}")

model.save_pretrained("model")
