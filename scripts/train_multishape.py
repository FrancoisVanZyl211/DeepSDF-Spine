import argparse
import numpy as np
import os, sys
import torch
import torch.backends.cudnn as cudnn
from model_multishape import MultiShapeDecoder
from utils import SdfDataset, normalize_pts, normalize_normals
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.train_settings import TrainSettings
import torch.optim as optim

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    combined_data = np.load(args.multi_shape_file) 
    xyz_data = combined_data[:, :3]
    normals_data = combined_data[:, 3:6]
    shape_id_data = combined_data[:, 6].astype(np.int32)
    xyz_data = normalize_pts(xyz_data)
    normals_data = normalize_normals(normals_data)

    N = xyz_data.shape[0]
    n_train = int(args.train_split_ratio * N)
    indices = np.arange(N)
    np.random.shuffle(indices)
    idx_train = indices[:n_train]
    idx_val = indices[n_train:]

    train_dataset = SdfDataset(
        points=xyz_data[idx_train],
        normals=normals_data[idx_train],
        shape_ids=shape_id_data[idx_train, None],
        phase='train',
        args=args
    )
    val_dataset = SdfDataset(
        points=xyz_data[idx_val],
        normals=normals_data[idx_val],
        shape_ids=shape_id_data[idx_val, None],
        phase='val',
        args=args
    )

    num_shapes = shape_id_data.max() + 1 
    model = MultiShapeDecoder(num_shapes, args, latent_dim=args.latent_dim).to(device)
    print(f"Number of shapes: {num_shapes}")
    print(f"Model param count: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        train_loss = run_epoch(train_dataset, model, optimizer, device, args, is_train=True)
        val_loss   = run_epoch(val_dataset,   model, optimizer, device, args, is_train=False)
        print(f"Epoch [{epoch+1}/{args.epochs}] - TrainLoss: {train_loss:.4f}, ValLoss: {val_loss:.4f}")

    if args.out_model is not None:
        checkpoint_path = args.out_model
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    else:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        checkpoint_dir = os.path.join(script_dir, "..", "data")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "multi_shape_deepsdf.pth")
    
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

def run_epoch(dataset, model, optimizer, device, args, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    num_batches = len(dataset)
    total_loss = 0.0
    sigma = 0.1 

    for i in range(num_batches):
        batch = dataset[i]
        xyz = batch['xyz'].to(device)
        sdf_gt = batch['gt_sdf'].to(device)
        shape_id = batch['shape_id'].to(device)

        if is_train:
            optimizer.zero_grad()

        sdf_pred = model(xyz, shape_id)
        sdf_pred_clamped = torch.clamp(sdf_pred, -sigma, sigma)
        sdf_gt_clamped   = torch.clamp(sdf_gt.view(-1, 1), -sigma, sigma)
        loss_tensor = torch.abs(sdf_pred_clamped - sdf_gt_clamped)
        loss = torch.sum(loss_tensor)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (num_batches * args.train_batch)
    return avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- regular CLI args ---
    parser.add_argument("--multi_shape_file", default="multi_shape_data.npy")
    parser.add_argument("--train_split_ratio", default=0.8, type=float)
    parser.add_argument("--latent_dim",  default=64,  type=int)
    parser.add_argument("--hidden_dim", default=512, type=int, help="Width of the MLP layers in MultiShapeDecoder")
    parser.add_argument("--dropout",     default=0.10, type=float)
    parser.add_argument("--lr",          default=1e-4, type=float)
    parser.add_argument("--weight_decay",default=1e-4, type=float)
    parser.add_argument("--epochs",      default=10,   type=int)
    parser.add_argument("--train_batch", default=512,  type=int)
    parser.add_argument("--sample_std",  default=0.05, type=float)
    parser.add_argument("--N_samples",   default=10,   type=int)
    parser.add_argument("--grid_N",      default=128,  type=int)
    parser.add_argument("--max_xyz",     default=1.0,  type=float)
    parser.add_argument("--out_model",   type=str, default=None)

    # --- JSON config ---
    parser.add_argument("--config", type=str,
                        help="Path to TrainSettings JSON (overrides defaults)")

    cfg_args, _ = parser.parse_known_args()

    if cfg_args.config:
        cfg = TrainSettings.from_json(cfg_args.config)
        parser.set_defaults(**cfg.__dict__)

    args = parser.parse_args()
    main(args)

    # Example usage:
    # python train_multishape.py --multi_shape_file "..\data\multi_shape_data.npy" --train_split_ratio 0.8 --latent_dim 64 --lr 1e-4 --weight_decay 1e-4 --epochs 80 --train_batch 512 --N_samples 10 --out_model "C:/my_chosen_folder/model.pth"