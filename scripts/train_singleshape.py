import argparse
import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.train_settings import TrainSettings

# Define a simple dataset for a single shape
class SingleShapeDataset(Dataset):
    def __init__(self, data_file):
        """
        Expects data_file to be a .npy file.
        If the file has 7 columns, we use:
          [x, y, z, sdf, ..., ...] (we assume SDF is in column 3).
        Otherwise, we create a dummy ground-truth SDF (all zeros).
        """
        self.data = np.load(data_file)
        if self.data.shape[1] >= 4:
            self.xyz = self.data[:, :3]
            self.gt_sdf = self.data[:, 3]
        else:
            self.xyz = self.data[:, :3]
            self.gt_sdf = np.zeros(self.data.shape[0], dtype=np.float32)

    def __len__(self):
        return len(self.xyz)

    def __getitem__(self, idx):
        sample = {
            'xyz': torch.tensor(self.xyz[idx], dtype=torch.float32),
            'gt_sdf': torch.tensor(self.gt_sdf[idx], dtype=torch.float32).unsqueeze(0)
        }
        return sample

class SingleShapeDecoder(nn.Module):
    def __init__(self, hidden_dim=512):
        super(SingleShapeDecoder, self).__init__()
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.tanh(x)

def train_epoch(dataloader, model, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        xyz = batch['xyz'].to(device)
        gt_sdf = batch['gt_sdf'].to(device)
        pred = model(xyz)
        loss = torch.mean(torch.abs(pred - gt_sdf))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_epoch(dataloader, model, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            xyz = batch['xyz'].to(device)
            gt_sdf = batch['gt_sdf'].to(device)
            pred = model(xyz)
            loss = torch.mean(torch.abs(pred - gt_sdf))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset = SingleShapeDataset(args.data_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = SingleShapeDecoder(hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    
    for epoch in range(args.epochs):
        train_loss = train_epoch(dataloader, model, optimizer, device)
        val_loss = eval_epoch(dataloader, model, device)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    output_path = args.out_model if args.out_model is not None else "single_shape_deepsdf.pth"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="single_shape.npy")
    parser.add_argument("--hidden_dim", default=512, type=int, help="Width of the MLP layers in MultiShapeDecoder")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--epochs",      default=10,   type=int)
    parser.add_argument("--batch_size",  default=512,  type=int)
    parser.add_argument("--lr",          default=1e-4, type=float)
    parser.add_argument("--out_model",   type=str, default=None)

    parser.add_argument("--config", type=str,
                        help="Path to TrainSettings JSON (overrides defaults)")

    cfg_args, _ = parser.parse_known_args()
    if cfg_args.config:
        cfg = TrainSettings.from_json(cfg_args.config)
        parser.set_defaults(**cfg.__dict__)

    args = parser.parse_args()
    main(args)