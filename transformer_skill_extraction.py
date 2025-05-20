import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os
from tqdm import tqdm
from seqeval.metrics import classification_report, accuracy_score, f1_score
from skill_extraction_system import SkillExtractor

# Cấu hình cho mô hình
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 3e-5
MODEL_NAME = "vinai/phobert-base"  # Sử dụng PhoBERT cho tiếng Việt
OUTPUT_DIR = "models/phobert_skill_extraction"
LABELS = ["O", "B-SKILL", "I-SKILL"]  # O: Không phải kỹ năng, B-SKILL: Bắt đầu kỹ năng, I-SKILL: Bên trong kỹ năng


class JobSkillDataset(Dataset):
    def __init__(self, texts, tags, tokenizer, max_len):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        tags = self.tags[item]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True
        )

        # Chuyển đổi nhãn thẻ sang ID
        label_ids = [LABELS.index("O")] * len(encoding["input_ids"])  # Mặc định là "O"

        # Ánh xạ nhãn thẻ vào token
        word_ids = encoding.word_ids()
        previous_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue

            if word_idx < len(tags):
                label_ids[idx] = LABELS.index(tags[word_idx])

            previous_word_idx = word_idx

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }


def create_training_data(job_data_path, annotations_path):
    """Tạo dữ liệu huấn luyện từ dữ liệu job và file annotation"""
    # Đọc dữ liệu job
    with open(job_data_path, 'r', encoding='utf-8') as f:
        jobs = json.load(f)

    # Đọc dữ liệu gán nhãn
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    texts = []
    tags_list = []

    for job_id, job_annotation in annotations.items():
        # Tìm job tương ứng
        job = None
        for j in jobs:
            if str(j.get("id")) == job_id:
                job = j
                break

        if job is None:
            continue

        requirements = job.get("requirements", "")
        if not requirements:
            continue

        # Phân tách requirements thành các từ
        words = requirements.split()

        # Tạo list các tags tương ứng
        tags = ["O"] * len(words)

        # Gán các nhãn theo annotation
        for skill_span in job_annotation.get("skills", []):
            start_idx = skill_span["start_word"]
            end_idx = skill_span["end_word"]

            if start_idx >= len(tags):
                continue

            # Gán nhãn B-SKILL cho từ đầu tiên của kỹ năng
            tags[start_idx] = "B-SKILL"

            # Gán nhãn I-SKILL cho các từ còn lại của kỹ năng
            for i in range(start_idx + 1, min(end_idx + 1, len(tags))):
                tags[i] = "I-SKILL"

        texts.append(requirements)
        tags_list.append(tags)

    return texts, tags_list


def train_model():
    """Huấn luyện mô hình trích xuất kỹ năng"""
    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Chuẩn bị dữ liệu
    texts, tags_list = create_training_data("data/jobs.json", "data/annotations.json")

    # Phân chia tập train/validation
    train_texts, val_texts, train_tags, val_tags = train_test_split(
        texts, tags_list, test_size=0.2, random_state=42
    )

    # Tải tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tạo dataset
    train_dataset = JobSkillDataset(train_texts, train_tags, tokenizer, MAX_LEN)
    val_dataset = JobSkillDataset(val_texts, val_tags, tokenizer, MAX_LEN)

    # Tạo dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Tải mô hình
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS)
    )

    # Đưa mô hình lên GPU nếu có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Chuẩn bị optimizer và scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Vòng lặp huấn luyện
    best_f1 = 0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        # Training
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")

        for batch in progress_bar:
            # Đưa dữ liệu lên thiết bị
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss}")

        # Evaluation
        model.eval()
        predictions = []
        true_labels = []

        for batch in tqdm(val_dataloader, desc="Evaluating"):
            # Đưa dữ liệu lên thiết bị
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # Lấy dự đoán
            preds = torch.argmax(outputs.logits, dim=2)

            # Chuyển đổi dự đoán và nhãn thật thành danh sách các nhãn
            for i in range(input_ids.shape[0]):
                pred_list = []
                true_list = []

                for j in range(input_ids.shape[1]):
                    # Bỏ qua các token đặc biệt và padding
                    if labels[i, j] != -100:
                        pred_list.append(LABELS[preds[i, j].item()])
                        true_list.append(LABELS[labels[i, j].item()])

                predictions.append(pred_list)
                true_labels.append(true_list)

        # Tính toán các chỉ số hiệu suất
        report = classification_report(true_labels, predictions)
        current_f1 = f1_score(true_labels, predictions)
        print(f"Validation F1 Score: {current_f1}")
        print(report)

        # Lưu mô hình tốt nhất
        if current_f1 > best_f1:
            best_f1 = current_f1
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"Đã lưu mô hình tốt nhất với F1: {best_f1}")

    print("Huấn luyện hoàn tất!")
    return model, tokenizer


class SkillExtractorTransformer:
    def __init__(self, model_path=None):
        """Khởi tạo bộ trích xuất kỹ năng dựa trên Transformer"""
        if model_path is None:
            model_path = OUTPUT_DIR

        # Tải mô hình và tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            print(f"Đã tải mô hình từ {model_path}")
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            print("Đang tải mô hình mặc định...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForTokenClassification.from_pretrained(
                MODEL_NAME, num_labels=len(LABELS)
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def extract_skills(self, text):
        """Trích xuất kỹ năng từ văn bản"""
        if not text:
            return []

        # Tokenize văn bản
        tokens = self.tokenizer(
            text,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Đưa dữ liệu lên thiết bị
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        # Dự đoán
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Lấy nhãn dự đoán
        predictions = torch.argmax(outputs.logits, dim=2)
        predictions = predictions[0].cpu().numpy()

        # Chuyển đổi dự đoán thành nhãn
        pred_labels = [LABELS[p] for p in predictions]

        # Tách văn bản thành các từ
        words = text.split()

        # Trích xuất kỹ năng
        skills = []
        current_skill = []

        for i, (word, offset) in enumerate(zip(words, tokens["offset_mapping"][0])):
            token_idx = offset[0].item()
            if token_idx >= len(pred_labels):
                break

            label = pred_labels[token_idx]

            if label == "B-SKILL":
                if current_skill:
                    skills.append(" ".join(current_skill))
                    current_skill = []
                current_skill.append(word)
            elif label == "I-SKILL" and current_skill:
                current_skill.append(word)
            elif label == "O" and current_skill:
                skills.append(" ".join(current_skill))
                current_skill = []

        # Thêm kỹ năng cuối cùng nếu có
        if current_skill:
            skills.append(" ".join(current_skill))

        return skills


def create_annotation_data(rule_based_results, output_file="data/annotations.json"):
    """Tạo dữ liệu annotation từ kết quả rule-based để huấn luyện mô hình"""
    annotations = {}

    for job in rule_based_results["job_analyses"]:
        job_id = job.get("job_id", str(hash(job["job_title"])))
        skills = job.get("extracted_skills", [])

        if not skills:
            continue

        requirements = job.get("requirements", "")
        if not requirements:
            continue

        # Tách yêu cầu thành các từ
        words = requirements.split()

        # Tìm vị trí của các kỹ năng trong văn bản
        skill_spans = []
        for skill in skills:
            skill_words = skill.split()
            # Xử lý trường hợp skill có thể trống
            if not skill_words:
                continue

            for i in range(len(words) - len(skill_words) + 1):
                # Kiểm tra xem skill có xuất hiện tại vị trí i không
                match = True
                for j in range(len(skill_words)):
                    if i + j >= len(words) or words[i + j].lower() != skill_words[j].lower():
                        match = False
                        break

                if match:
                    skill_spans.append({
                        "skill": skill,
                        "start_word": i,
                        "end_word": i + len(skill_words) - 1
                    })

        annotations[job_id] = {
            "requirements": requirements,
            "skills": skill_spans
        }

    # Lưu annotations ra file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)

    print(f"Đã tạo dữ liệu annotation cho {len(annotations)} jobs tại {output_file}")
    return annotations


def main():
    """Quy trình đầy đủ: từ rule-based đến transformer-based"""
    # 1. Tạo thư mục cho dữ liệu và mô hình
    os.makedirs("data", exist_ok=True)

    # 2. Sử dụng rule-based extractor để tạo dữ liệu huấn luyện ban đầu
    extractor = SkillExtractor()

    # Giả sử đã có file jobs.json chứa dữ liệu job
    with open("data/jobs.json", 'r', encoding='utf-8') as f:
        jobs = json.load(f)

    # Phân tích dữ liệu job bằng phương pháp rule-based
    rule_based_results = extractor.analyze_job_batch(jobs)

    # 3. Tạo dữ liệu annotation từ kết quả rule-based
    create_annotation_data(rule_based_results)

    # 4. Huấn luyện mô hình transformer
    print("Bắt đầu huấn luyện mô hình transformer...")
    model, tokenizer = train_model()

    # 5. Sử dụng mô hình đã huấn luyện để trích xuất kỹ năng
    transformer_extractor = SkillExtractorTransformer()

    # 6. So sánh kết quả giữa rule-based và transformer-based
    print("\nSo sánh kết quả trích xuất kỹ năng:")

    for i, job in enumerate(jobs[:5]):  # Lấy 5 job đầu tiên để so sánh
        requirements = job.get("requirements", "")
        if not requirements:
            continue

        print(f"\nJob {i + 1}: {job.get('title', '')}")
        print("Requirements:", requirements[:100] + "..." if len(requirements) > 100 else requirements)

        # Kết quả từ rule-based
        rule_skills = extractor.extract_skills_from_job(requirements)
        print("\nRule-based skills:", rule_skills)

        # Kết quả từ transformer-based
        transformer_skills = transformer_extractor.extract_skills(requirements)
        print("Transformer-based skills:", transformer_skills)

    print("\nQuá trình hoàn tất!")


if __name__ == "__main__":
    main()