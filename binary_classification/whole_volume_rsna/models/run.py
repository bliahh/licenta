import torch
from tqdm import tqdm



def run(model, loader, criterion, scaler, optimizer, device, use_amp, desc="train", accum_steps=2):
    model.train()
    total = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    bar = tqdm(loader, desc=desc, leave=False)
    for step, (x, y) in enumerate(bar, start=1):
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = criterion(model(x), y)

        #scaler.scale(loss).backward()
        scaler.scale(loss / accum_steps).backward()

        if step % accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total += loss.item() * len(y)
        n += len(y)
        bar.set_postfix(loss=f"{total / n:.4f}")

    return total / n



@torch.no_grad()
def predict(model, loader, device, use_amp, desc="val"):
    model.eval()
    logits, labels = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()