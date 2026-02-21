import torch
from torch.utils.data import DataLoader
from dataset import SRDataset
from hat_arch import HAT
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = SRDataset(
    r"C:\work\CVPR 2026\wYe7pBJ7-train\train",
    hr_size=(32, 64),
    scale=2
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = HAT(
    img_size=(16,32),
    upscale=2,
    upsampler='pixelshuffle',
    window_size=8
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
criterion = torch.nn.L1Loss()

best_loss = float("inf")

for epoch in range(50):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/50")

    for lr, hr in pbar:
        lr = lr.to(device)
        hr = hr.to(device)

        sr = model(lr)
        loss = criterion(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}: {avg_loss:.4f}")

    # 🔥 save last
    torch.save(model.state_dict(), "last_model.pth")

    # 🔥 save best
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model!")
