import pandas as pd
import torch.utils.data.distributed
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
from Utils.utils import set_seed, create_dir, setup_logger, AvgMeter, print_and_log_output
from utils_data.dataset import build_train_val_loaders, calculate_st_embedding
from tricks.lr.lr_scheduler import WarmupCosineLR
import time
import glob

from config import get_config
from model.models import CLIPModel_ViT_itm_v14_MSE


# 训练过程
def train_one_epoch(epoch, model, data_loader, optimizer, scaler, ema_model, accumulation_steps, print_seq, logger):
    model.train()
    start_time = time.time()
    train_loss_meter = AvgMeter('Training Loss')
    train_loss_1_meter = AvgMeter('Training Loss_1')
    train_loss_2_meter = AvgMeter('Training Loss_2')
    pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Training]")

    for batch_idx, inputs in enumerate(pbar):
        inputs = {k: v.cuda() for k, v in inputs.items() if k == "image" or k == "st_data"}
        with autocast():
            loss_1, loss_2 = model(inputs)
            loss = loss_1 + loss_2

        batch_loss = loss / accumulation_steps
        scaler.scale(batch_loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(data_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        with torch.no_grad():
            for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data = ema_param.data * 0.99 + model_param.data * 0.01

        train_loss_meter.update(loss.item())
        train_loss_1_meter.update(loss_1.item())
        train_loss_2_meter.update(loss_2.item())
        pbar.set_postfix({
            "Training Epoch": epoch + 1,
            "Batch": f"{batch_idx + 1}/{len(data_loader)}",
            "Loss": f"{loss.item():.4f}",
            "Average Loss": f"{train_loss_meter.avg:.4f}",
            "Loss_1": f"{train_loss_1_meter.avg:.4f}",
            "Loss_2": f"{train_loss_2_meter.avg:.4f}",
        })
        if (batch_idx + 1) % print_seq == 0 or (batch_idx + 1) == len(data_loader):
            logger.info(
                f"Training Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(data_loader)} - "
                f"Loss: {loss.item():.4f}, Average Loss: {train_loss_meter.avg:.4f},"
                f"Loss_1: {train_loss_1_meter.avg:.4f}, Loss_2: {train_loss_2_meter.avg:.4f}")

    end_time = time.time()
    epoch_time = end_time - start_time
    return train_loss_meter, epoch_time


def validate(epoch, model, data_loader, print_seq, logger):
    model.eval()
    val_loss_meter = AvgMeter('Valid Loss')
    train_loss_1_meter = AvgMeter('Training Loss_1')
    train_loss_2_meter = AvgMeter('Training Loss_2')
    start_time = time.time()
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Validation]")
        for batch_idx, inputs in enumerate(pbar):
            inputs = {k: v.cuda() for k, v in inputs.items() if k == "image" or k == "st_data"}
            with autocast():
                loss_1, loss_2 = model(inputs)
                loss = loss_1 + loss_2

            val_loss_meter.update(loss.item())
            train_loss_1_meter.update(loss_1.item())
            train_loss_2_meter.update(loss_2.item())

            pbar.set_postfix({
                "Validating Epoch": epoch + 1,
                "Batch": f"{batch_idx + 1}/{len(data_loader)}",
                "Loss": f"{loss.item():.4f}",
                "Average Loss": f"{val_loss_meter.avg:.4f}",
                "Loss_1": f"{train_loss_1_meter.avg:.4f}",
                "Loss_2": f"{train_loss_2_meter.avg:.4f}",
            })

            if (batch_idx + 1) % print_seq == 0 or (batch_idx + 1) == len(data_loader):
                logger.info(
                    f"Training Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(data_loader)} - "
                    f"Loss: {loss.item():.4f}, Average Loss: {val_loss_meter.avg:.4f}"
                    f"Loss_1: {train_loss_1_meter.avg:.4f}, Loss_2: {train_loss_2_meter.avg:.4f}")

    end_time = time.time()
    epoch_time = end_time - start_time
    return val_loss_meter, epoch_time


def save_model(model, optimizer, epoch, save_dir, val_loss, filename="model.pth"):
    save_path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, save_path)


if __name__ == "__main__":
    args = get_config()
    set_seed(42)

    embedding_length = calculate_st_embedding(args=args)

    for cancer in args.cancer_list:
        logger = setup_logger(cancer=cancer)

        split_file_list = glob.glob(os.path.join(args.data_path, str(cancer), 'splits', '*.csv'))
        k = int(len(split_file_list) / 2)
        for i in range(k):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            save_dir = os.path.join(args.save_dir, cancer, 'pth_' + str(i))
            create_dir(save_dir)
            effective_batch_size = args.batch_size * args.accumulation_steps

            model = CLIPModel_ViT_itm_v14_MSE(spot_embedding=embedding_length[str(cancer)]).to(device)
            ema_model = CLIPModel_ViT_itm_v14_MSE(spot_embedding=embedding_length[str(cancer)]).to(device)
            ema_model.load_state_dict(model.state_dict())

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = WarmupCosineLR(optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.num_epochs,
                                       min_lr=args.min_lr, max_lr=args.max_lr)

            scaler = GradScaler()
            split_df = pd.read_csv(os.path.join(args.data_path, cancer, 'splits', 'train_' + str(i) + '.csv'))
            sample_list = list(split_df.iloc[:]['sample_id'])
            train_loader, val_loader = build_train_val_loaders(
                args,
                cancer=cancer,
                sample_list=sample_list,
            )
            print_and_log_output(logger, optimizer, args, effective_batch_size, device)

            best_val_loss = float('inf')
            for epoch in range(args.num_epochs):
                current_lr = scheduler.step(epoch)

                train_loss, train_time = train_one_epoch(epoch, model, train_loader, optimizer, scaler, ema_model,
                                                         args.accumulation_steps, args.print_seq, logger)
                logger.info(
                    f"Epoch [{epoch + 1}/{args.num_epochs}], Training Loss: {train_loss.avg:.4f}, Training Time: {train_time:.2f}s, Learning Rate: {current_lr:.6f}")

                val_loss, val_time = validate(epoch, model, val_loader, args.print_seq, logger)
                logger.info(
                    f"Epoch [{epoch + 1}/{args.num_epochs}], Validation Loss: {val_loss.avg:.4f}, Validation Time: {val_time:.2f}s")

                if val_loss.avg < best_val_loss:
                    best_val_loss = val_loss.avg
                    save_model(model, optimizer, epoch, save_dir, val_loss.avg, filename="best_model.pth")
                    logger.info(f"Saved best model at epoch {epoch + 1}")

                if epoch > 30 and (epoch + 1) % 10 == 0:
                    save_model(model, optimizer, epoch, save_dir, val_loss, filename=f"model_epoch_{epoch + 1}.pth")
