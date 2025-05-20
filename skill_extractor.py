import os
import re
import json
import jsonlines
import spacy
import pandas as pd
import numpy as np
from collections import Counter
from skill_dictionaries import SOFT_SKILLS, HARD_SKILLS
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Đảm bảo đã tải các tài nguyên cần thiết
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class SkillExtractor:
    def __init__(self, use_spacy=True):
        # Tạo từ điển kỹ năng
        self.soft_skills = set(SOFT_SKILLS)
        self.hard_skills = set(HARD_SKILLS)
        self.all_skills = self.soft_skills.union(self.hard_skills)

        # Các mẫu regex để tìm kiếm kỹ năng
        self.skill_patterns = [
            r"có kỹ năng ([\w\s]+)",
            r"yêu cầu ([\w\s]+)",
            r"ưu tiên ([\w\s]+)",
            r"có khả năng ([\w\s]+)",
            r"có kinh nghiệm ([\w\s]+)",
            r"có kiến thức ([\w\s]+)",
            r"thành thạo ([\w\s]+)",
            r"sử dụng thành thạo ([\w\s]+)",
            r"sử dụng ([\w\s]+)",
            r"biết ([\w\s]+)",
            r"nắm vững ([\w\s]+)",
            r"tiếng ([\w\s]+)",
            r"có tư duy ([\w\s]+)"
        ]

        # Các từ khóa chỉ thị kỹ năng
        self.skill_indicators = [
            "kỹ năng", "khả năng", "kinh nghiệm", "kiến thức",
            "thành thạo", "sử dụng", "biết", "nắm vững", "tư duy",
            "trình độ", "chuyên môn", "năng lực", "tiếng", "ngôn ngữ"
        ]

        # Stopwords tiếng Việt
        vietnamese_stopwords = set(stopwords.words('vietnamese') if 'vietnamese' in stopwords._fileids else [])
        custom_stopwords = {"có", "và", "với", "các", "để", "từ", "trở", "lên", "trong", "hoặc", "như", "là", "được",
                            "những"}
        self.stopwords = vietnamese_stopwords.union(custom_stopwords)

        # Tải NLP model nếu cần
        self.use_spacy = use_spacy
        if use_spacy:
            try:
                self.nlp = spacy.load("vi_core_news_lg")
                print("Đã tải mô hình tiếng Việt thành công")
            except:
                try:
                    # Sử dụng mô hình tiếng Anh nếu không có mô hình tiếng Việt
                    self.nlp = spacy.load("en_core_web_sm")
                    print("Không tìm thấy mô hình tiếng Việt. Sử dụng mô hình tiếng Anh thay thế.")
                except:
                    self.use_spacy = False
                    print("Không thể tải mô hình spaCy. Sẽ chỉ sử dụng phương pháp rule-based.")

    def preprocess_text(self, text):
        """Tiền xử lý văn bản"""
        if not text or not isinstance(text, str):
            return ""

        # Chuyển thành chữ thường
        text = text.lower()

        # Loại bỏ các ký tự đặc biệt
        text = re.sub(r'[^\w\s\-\+/]', ' ', text)

        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_skills_by_dictionary(self, text):
        """Trích xuất kỹ năng bằng cách so khớp từ điển"""
        text = self.preprocess_text(text)
        found_skills = {'soft_skills': [], 'hard_skills': []}

        # Tìm kiếm kỹ năng mềm
        for skill in self.soft_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text):
                found_skills['soft_skills'].append(skill)

        # Tìm kiếm kỹ năng chuyên môn
        for skill in self.hard_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text):
                found_skills['hard_skills'].append(skill)

        return found_skills

    def extract_skills_by_pattern(self, text):
        """Trích xuất kỹ năng bằng cách sử dụng các mẫu regex"""
        text = self.preprocess_text(text)
        potential_skills = []

        # Tìm kiếm theo mẫu
        for pattern in self.skill_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                skill_text = match.group(1).strip()
                if skill_text and len(skill_text.split()) <= 5:  # Giới hạn độ dài kỹ năng
                    potential_skills.append(skill_text)

        # Tìm kiếm câu chứa từ khóa chỉ thị
        sentences = sent_tokenize(text)
        for sentence in sentences:
            for indicator in self.skill_indicators:
                if indicator in sentence:
                    # Lấy cụm từ sau từ khóa chỉ thị
                    parts = sentence.split(indicator)
                    if len(parts) > 1:
                        potential_skill = parts[1].strip()
                        if potential_skill and len(potential_skill.split()) <= 5:
                            potential_skills.append(potential_skill)

        # Phân loại các kỹ năng tiềm năng
        found_skills = {'soft_skills': [], 'hard_skills': []}

        for skill in potential_skills:
            # Kiểm tra xem kỹ năng có trong từ điển không
            if skill in self.soft_skills:
                found_skills['soft_skills'].append(skill)
            elif skill in self.hard_skills:
                found_skills['hard_skills'].append(skill)
            else:
                # Nếu không có trong từ điển, kiểm tra các từ trong kỹ năng
                words = skill.split()
                for word in words:
                    if word in self.soft_skills:
                        found_skills['soft_skills'].append(word)
                    elif word in self.hard_skills:
                        found_skills['hard_skills'].append(word)

        return found_skills

    def extract_skills_by_nlp(self, text):
        """Trích xuất kỹ năng sử dụng NLP (nếu có)"""
        if not self.use_spacy:
            return {'soft_skills': [], 'hard_skills': []}

        text = self.preprocess_text(text)
        potential_skills = []

        # Phân tích cú pháp
        doc = self.nlp(text)

        # Trích xuất danh từ và cụm danh từ tiềm năng là kỹ năng
        for chunk in doc.noun_chunks:
            # Loại bỏ stopwords
            filtered_chunk = ' '.join([token.text for token in chunk if token.text not in self.stopwords])
            if filtered_chunk and len(filtered_chunk.split()) <= 5:
                potential_skills.append(filtered_chunk)

        # Trích xuất từ dựa trên POS tagging
        for token in doc:
            if token.pos_ in ["NOUN", "ADJ"] and token.text not in self.stopwords:
                potential_skills.append(token.text)

        # Phân loại các kỹ năng tiềm năng
        found_skills = {'soft_skills': [], 'hard_skills': []}

        for skill in potential_skills:
            if skill in self.soft_skills:
                found_skills['soft_skills'].append(skill)
            elif skill in self.hard_skills:
                found_skills['hard_skills'].append(skill)
            else:
                # Nếu không có trong từ điển, kiểm tra các từ trong kỹ năng
                words = skill.split()
                for word in words:
                    if word in self.soft_skills:
                        found_skills['soft_skills'].append(word)
                    elif word in self.hard_skills:
                        found_skills['hard_skills'].append(word)

        return found_skills

    def extract_skills(self, text):
        """Trích xuất kỹ năng từ văn bản, kết hợp tất cả các phương pháp"""
        if not text:
            return {'soft_skills': [], 'hard_skills': []}

        # Kết hợp các phương pháp trích xuất
        dict_skills = self.extract_skills_by_dictionary(text)
        pattern_skills = self.extract_skills_by_pattern(text)
        nlp_skills = self.extract_skills_by_nlp(text) if self.use_spacy else {'soft_skills': [], 'hard_skills': []}

        # Gộp kết quả
        all_soft_skills = set(dict_skills['soft_skills'] + pattern_skills['soft_skills'] + nlp_skills['soft_skills'])
        all_hard_skills = set(dict_skills['hard_skills'] + pattern_skills['hard_skills'] + nlp_skills['hard_skills'])

        # Loại bỏ kỹ năng trùng lặp hoặc quá ngắn
        final_soft_skills = [skill for skill in all_soft_skills if len(skill) > 2]
        final_hard_skills = [skill for skill in all_hard_skills if len(skill) > 2]

        return {
            'soft_skills': sorted(list(final_soft_skills)),
            'hard_skills': sorted(list(final_hard_skills))
        }

    def process_jsonl_file(self, input_file, output_file):
        """Xử lý file JSONL và lưu kết quả"""
        processed_data = []

        # Đọc file JSONL
        with jsonlines.open(input_file) as reader:
            for item in reader:
                text = item.get('text', '')

                # Trích xuất kỹ năng
                skills = self.extract_skills(text)

                # Thêm kỹ năng vào dữ liệu
                item['labels'] = {
                    'soft_skills': skills['soft_skills'],
                    'hard_skills': skills['hard_skills']
                }

                processed_data.append(item)

        # Lưu kết quả
        with jsonlines.open(output_file, 'w') as writer:
            for item in processed_data:
                writer.write(item)

        print(f"Đã xử lý và lưu kết quả vào {output_file}")
        return processed_data

    def evaluate_extraction(self, processed_data):
        """Đánh giá kết quả trích xuất kỹ năng"""
        total_items = len(processed_data)
        items_with_skills = 0
        soft_skill_counts = Counter()
        hard_skill_counts = Counter()

        for item in processed_data:
            soft_skills = item.get('labels', {}).get('soft_skills', [])
            hard_skills = item.get('labels', {}).get('hard_skills', [])

            if soft_skills or hard_skills:
                items_with_skills += 1

            for skill in soft_skills:
                soft_skill_counts[skill] += 1

            for skill in hard_skills:
                hard_skill_counts[skill] += 1

        # Tính toán thống kê
        avg_soft_skills = sum(
            len(item.get('labels', {}).get('soft_skills', [])) for item in processed_data) / total_items
        avg_hard_skills = sum(
            len(item.get('labels', {}).get('hard_skills', [])) for item in processed_data) / total_items

        # In kết quả
        print(f"\n==== Thống kê kết quả trích xuất kỹ năng ====")
        print(f"Tổng số mẫu: {total_items}")
        print(f"Số mẫu có kỹ năng: {items_with_skills} ({items_with_skills / total_items * 100:.2f}%)")
        print(f"Số kỹ năng mềm trung bình mỗi mẫu: {avg_soft_skills:.2f}")
        print(f"Số kỹ năng chuyên môn trung bình mỗi mẫu: {avg_hard_skills:.2f}")

        print("\nTop 10 kỹ năng mềm phổ biến nhất:")
        for skill, count in soft_skill_counts.most_common(10):
            print(f"  - {skill}: {count} lần ({count / total_items * 100:.2f}%)")

        print("\nTop 10 kỹ năng chuyên môn phổ biến nhất:")
        for skill, count in hard_skill_counts.most_common(10):
            print(f"  - {skill}: {count} lần ({count / total_items * 100:.2f}%)")

        return {
            'total_items': total_items,
            'items_with_skills': items_with_skills,
            'avg_soft_skills': avg_soft_skills,
            'avg_hard_skills': avg_hard_skills,
            'top_soft_skills': soft_skill_counts.most_common(10),
            'top_hard_skills': hard_skill_counts.most_common(10)
        }


# Hàm main để chạy thử
def main():
    # Đường dẫn file
    input_file = "./data/requirements_data.jsonl"  # Thay đổi đường dẫn phù hợp với file của bạn
    output_file = "job_requirements_with_skills.jsonl"

    # Khởi tạo extractor
    extractor = SkillExtractor(use_spacy=True)  # Set False nếu không muốn sử dụng spaCy

    # Xử lý file
    processed_data = extractor.process_jsonl_file(input_file, output_file)

    # Đánh giá kết quả
    evaluation = extractor.evaluate_extraction(processed_data)

    # Hiển thị một số mẫu
    print("\n==== Ví dụ trích xuất kỹ năng ====")
    for i, item in enumerate(processed_data[:5]):  # Hiển thị 5 mẫu đầu tiên
        print(f"\nMẫu {i + 1}:")
        print(f"Text: {item['text'][:200]}...")
        print(f"Kỹ năng mềm: {', '.join(item['labels']['soft_skills'])}")
        print(f"Kỹ năng chuyên môn: {', '.join(item['labels']['hard_skills'])}")
        print("-" * 50)


if __name__ == "__main__":
    main()
