import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm

def setup_ddp(rank, world_size):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup_ddp(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'AIDER'))
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 4)  # Adjust if needed
    model = model.to(device)
    model = DDP(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(2):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"[Rank {rank}] Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        if rank == 0:
            print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}, Accuracy: {correct/total:.4f}")

    if rank == 0:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ader_ddp_cpu.pt')
        torch.save(model.module.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")

    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()

    main(args.rank, args.world_size)

