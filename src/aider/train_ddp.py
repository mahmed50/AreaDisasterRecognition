import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
import time, json, platform
import torch.distributed as dist

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
    VAL_DIR = os.path.join(DATA_DIR, 'val')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    
    MAX_SAMPLES = 5000  # or 200 for ultra-fast tests
    train_dataset = torch.utils.data.Subset(train_dataset, range(MAX_SAMPLES))
    val_dataset = torch.utils.data.Subset(val_dataset, range(1000))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 4)  # Adjust if needed
    model = model.to(device)
    model = DDP(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    results = {
    "system": platform.node(),
    "device": str(device),
    "model": "resnet18",
    "epochs": 10,
    "start_time": time.time(),
    "metrics": []
    }

    for epoch in range(10):
        epoch_start = time.time()
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

        
        correct_tensor = torch.tensor(correct, dtype=torch.float, device=device)
        total_tensor = torch.tensor(total, dtype=torch.float, device=device)

        # Sum across all processes
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

        # Global accuracy (same as single-node)
        global_acc = correct_tensor.item() / total_tensor.item()

        if rank == 0:
            print(f"[Epoch {epoch+1}] Global Accuracy: {global_acc:.4f}")
            epoch_time = (time.time() - epoch_start) / 60
            
            results["metrics"].append({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_acc": global_acc,
                "epoch_time_min": epoch_time
            })

    # Each rank evaluates independently
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i % world_size != rank:
                continue  # skip batches not assigned to this rank

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    # Convert to tensors and all-reduce
    correct_tensor = torch.tensor(correct, dtype=torch.float, device=device)
    total_tensor = torch.tensor(total, dtype=torch.float, device=device)

    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    val_acc = correct_tensor.item() / total_tensor.item()

    if rank == 0:
        print(f"✅ Global Validation Accuracy: {val_acc:.4f}")
        results["val_acc"] = val_acc

    if rank == 0:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ader_ddp_cpu.pt')
        torch.save(model.module.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")

    if rank == 0 or not hasattr(dist, "is_initialized") or not dist.is_initialized():
        out_file = os.path.join(os.path.dirname(__file__), '..', 'metrics', 'metrics_ddp.json')
        with open(
            out_file, 'w'
        ) as f:
            json.dump(results, f, indent=2)
        print(f"📈 Metrics saved to {out_file}")


    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()

    main(args.rank, args.world_size)

