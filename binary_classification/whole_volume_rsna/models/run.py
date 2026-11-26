import torch
from tqdm import tqdm
import numpy as np


def run(model, loader, criterion, scaler, optimizer, device, use_amp, desc="train", accum_steps=2, heatmap_loss=None, lambd=1.0, pos_weight=50.0):
    model.train()
    total_loss, total_cls, total_heat, n = 0.0, 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    bar = tqdm(loader, desc=desc, leave=False)
    for step, batch in enumerate(bar, start=1):
        volume, label = batch[0].to(device), batch[1].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            if heatmap_loss is None:
                loss_cls = criterion(model(volume), label)
                loss_heat = torch.zeros((), device=device)
            else:
                heat_target, has_target = batch[2].to(device), batch[3].to(device)
                logit, heat_map = model(volume)
                loss_cls = criterion(logit, label)
                loss_heat = heatmap_loss(heat_map, heat_target, has_target, pos_weight)
            loss = loss_cls + lambd * loss_heat

        scaler.scale(loss / accum_steps).backward()

        if step % accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = len(label)
        total_loss += loss.item() * batch_size
        total_cls += loss_cls.item() * batch_size
        total_heat += loss_heat.item() * batch_size
        n += batch_size
        bar.set_postfix(cls=f"{total_cls / n:.4f}", heat=f"{total_heat / n:.4f}")

    return total_loss / n, total_cls / n, total_heat / n


@torch.no_grad()
def predict(model, loader, device, use_amp, desc="predict", return_peaks=False, return_ids=False):
    model.eval()
    all_logits, all_labels, all_peaks, all_uids, all_slices = [], [], [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        volume = batch[0].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = model(volume)
        if isinstance(output, tuple):
            logit, heat_map = output
            any_map = heat_map[:, :1]
            flat_index = any_map.flatten(1).argmax(dim=1).cpu().numpy()
            all_peaks.append(np.stack(np.unravel_index(flat_index, any_map.shape[2:]), axis=1))
        else:
            logit = output
        logit = logit.float().cpu().numpy()
        label = batch[1].numpy()
        all_logits.append(logit.reshape(len(volume), -1))
        all_labels.append(label.reshape(len(volume), -1))
        if return_ids:
            all_uids.extend(batch[-2])
            all_slices.extend(batch[-1].numpy())
    logits, labels = np.concatenate(all_logits), np.concatenate(all_labels)
    if logits.shape[1] == 1:
        logits, labels = logits[:, 0], labels[:, 0]
    peaks = np.concatenate(all_peaks) if all_peaks else None
    if return_ids:
        return logits, labels, np.array(all_uids), np.array(all_slices), peaks
    if return_peaks:
        return logits, labels, peaks
    return logits, labels