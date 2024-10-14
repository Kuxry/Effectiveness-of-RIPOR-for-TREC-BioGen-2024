import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import csv
import sys
import wandb
from tqdm import tqdm
import torch.nn.utils as nn_utils

# 增加字段大小限制
csv.field_size_limit(sys.maxsize)


class PubMedDataset(Dataset):
    def __init__(self, tokenizer, data_path, chunk_size=1000, max_length=500):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.data = []
        self.current_chunk_start = 0
        self.current_chunk_end = 0
        self.total_len = self.calculate_total_length()

    def open_file(self):
        return open(self.data_path, 'r', encoding='utf-8')

    def load_next_chunk(self):
        with self.open_file() as file:
            tsv_reader = csv.reader(file, delimiter='\t')
            file.seek(self.current_chunk_start)
            self.data = []
            for _ in range(self.chunk_size):
                try:
                    row = next(tsv_reader)
                    if len(row) < 2:
                        continue
                    query_id = row[0].strip()
                    query_text = row[1].strip()
                    self.data.append({"query_id": query_id, "query": query_text})
                except (StopIteration, UnicodeDecodeError):
                    continue
            self.current_chunk_end = self.current_chunk_start + len(self.data)

    def calculate_total_length(self):
        with self.open_file() as file:
            total_len = sum(1 for _ in csv.reader(file, delimiter='\t'))
        return total_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx >= self.current_chunk_end or idx < self.current_chunk_start:
            self.current_chunk_start = idx
            self.load_next_chunk()

        idx_in_chunk = idx - self.current_chunk_start
        item = self.data[idx_in_chunk]
        input_text = item['query']
        input_encodings = self.tokenizer(
            input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": input_encodings["input_ids"].squeeze(),
            "attention_mask": input_encodings["attention_mask"].squeeze(),
            "labels": input_encodings["input_ids"].squeeze()
        }


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, model, tokenizer, train_dataset, val_dataset, output_dir, epochs=1, batch_size=12,
          learning_rate=1e-4, max_grad_norm=0.5):
    setup(rank, world_size)

    # 初始化 WandB
    if rank == 0:
        wandb.init(project="finetune", name=f"fine-training-{rank}")
        wandb.watch(model, log="all")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 配置学习率调度器
    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.1 * total_steps)  # warmup 步数为总步数的 10%
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # scaler = GradScaler()  # 取消混合精度训练

    for epoch in range(epochs):
        total_train_loss = 0
        model.train()

        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", position=rank)

        for batch_idx, batch in enumerate(train_loader_iter):
            torch.cuda.empty_cache()
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['labels'].to(rank)

            optimizer.zero_grad()

            # 不使用混合精度
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss is NaN or Inf at epoch {epoch + 1}, batch {batch_idx}. Skipping this batch.")
                continue

            loss.backward()

            # 梯度剪裁
            nn_utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            nan_found = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"NaN detected in gradients of {name}")
                        nan_found = True
                        break

            if nan_found:
                print("Skipping update due to NaN in gradients.")
                continue

            optimizer.step()
            scheduler.step()  # 在 optimizer.step() 之后调用

            total_train_loss += loss.item()

            if batch_idx % 30 == 0 and rank == 0:
                learning_rate = scheduler.get_last_lr()[0]
                epoch_fraction = epoch + batch_idx / len(train_loader)
                print({
                    'loss': loss.item(),
                    'learning_rate': learning_rate,
                    'rank_loss': loss.item(),  # Assuming rank_loss is the same as loss in this context
                    'epoch': epoch_fraction
                })
                wandb.log({
                    "Train Loss": loss.item(),
                    "Learning Rate": learning_rate,
                    "Epoch Fraction": epoch_fraction
                })
                train_loader_iter.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}")
            wandb.log({"Train Loss": avg_train_loss, "Epoch": epoch + 1})

        # 进行验证
        model.eval()
        total_val_loss = 0

        val_loader_iter = tqdm(val_loader, desc=f"Validation {epoch + 1}/{epochs}", position=rank)

        with torch.no_grad():
            for batch in val_loader_iter:
                torch.cuda.empty_cache()
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)
                labels = batch['labels'].to(rank)

                # 不使用混合精度
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")
            wandb.log({"Validation Loss": avg_val_loss, "Epoch": epoch + 1})  # 记录Validation Loss

    if rank == 0:
        model.module.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        wandb.save(os.path.join(output_dir, "pytorch_model.bin"))

    cleanup()


if __name__ == "__main__":
    world_size = 2

    model_checkpoint = "experiments-full-t5seq-aq/t5seq_aq_encoder_seq2seq_1_lng_knp_self_mnt_32_dcy_2/checkpoint"
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

    train_data_path = "../jsontotsv/s_train.tsv"
    val_data_path = "../jsontotsv/s_val.tsv"

    print("Loading datasets...")
    train_dataset = PubMedDataset(tokenizer, train_data_path)
    val_dataset = PubMedDataset(tokenizer, val_data_path)

    output_dir = "finemodel"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.multiprocessing.spawn(
        train,
        args=(world_size, model, tokenizer, train_dataset, val_dataset, output_dir),
        nprocs=world_size,
        join=True
    )
    if dist.get_rank() == 0:
        wandb.finish()
