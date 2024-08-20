import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from transformers import get_linear_schedule_with_warmup
import os


class PubMedDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length=512):
        self.tokenizer = tokenizer
        self.data = self.load_data(data_path)
        self.max_length = max_length

    def load_data(self, path):
        # 读取数据文件并解析为列表
        with open(path, 'r') as file:
            lines = file.readlines()
        data = [eval(line) for line in lines]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['query']
        target_text = item['doc']  # 假设doc是文档的摘要或内容
        input_encodings = self.tokenizer(
            input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        target_encodings = self.tokenizer(
            target_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": input_encodings["input_ids"].squeeze(),
            "attention_mask": input_encodings["attention_mask"].squeeze(),
            "labels": target_encodings["input_ids"].squeeze(),
        }


def train(model, tokenizer, train_dataset, val_dataset, output_dir, epochs=3, batch_size=8, learning_rate=3e-5,
          max_grad_norm=1.0):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        total_train_loss = 0
        model.train()

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}")

        # Validation step (optional)
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练模型和分词器
    model_checkpoint = "experiments-full-t5seq-aq/t5seq_aq_encoder_seq2seq_1_lng_knp_self_mnt_32_dcy_2/checkpoint"  # 替换为你的模型路径
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

    # 准备数据集
    train_data_path = "path_to_pubmed_train_data.txt"  # 替换为你的训练数据路径
    val_data_path = "path_to_pubmed_val_data.txt"  # 替换为你的验证数据路径

    train_dataset = PubMedDataset(tokenizer, train_data_path)
    val_dataset = PubMedDataset(tokenizer, val_data_path)

    # 输出路径
    output_dir = "finemodel"  # 替换为你希望保存微调模型的路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 开始训练
    train(model, tokenizer, train_dataset, val_dataset, output_dir)
