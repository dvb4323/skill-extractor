import pandas as pd
import re
import json
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
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
    def __init__(self):
        # Từ điển kỹ năng phổ biến (cần mở rộng)
        self.skill_dictionary = [
            "giảng dạy", "đào tạo", "truyền đạt", "nhanh nhẹn", "nhiệt tình",
            "tận tâm", "tiếng anh", "giao tiếp", "thuyết trình", "làm việc nhóm",
            "kỹ năng mềm", "powerpoint", "excel", "word", "office",
            "quản lý", "lãnh đạo", "photoshop", "illustrator", "thiết kế",
            "kiến thức về da", "chăm sóc da", "thẩm mỹ", "vật lý trị liệu"
        ]

        # Các mẫu regex để tìm kiếm kỹ năng
        self.skill_patterns = [
            r"có kỹ năng ([\w\s]+)",
            r"yêu cầu ([\w\s]+)",
            r"ưu tiên ([\w\s]+)",
            r"có khả năng ([\w\s]+)",
            r"có kinh nghiệm ([\w\s]+)",
            r"có kiến thức ([\w\s]+)"
        ]

        # Các từ khóa chỉ thị kỹ năng
        self.skill_indicators = [
            "kỹ năng", "khả năng", "kinh nghiệm", "kiến thức",
            "thành thạo", "sử dụng", "biết", "nắm vững"
        ]

        # Stopwords tiếng Việt
        vietnamese_stopwords = set(stopwords.words('vietnamese') if 'vietnamese' in stopwords._fileids else [])
        custom_stopwords = {"có", "và", "với", "các", "để", "từ", "trở", "lên", "trong"}
        self.stopwords = vietnamese_stopwords.union(custom_stopwords)

        # Cố gắng tải mô hình tiếng Việt nếu có
        try:
            import spacy
            self.nlp = spacy.load("vi_core_news_lg")
            print("Đã tải mô hình tiếng Việt thành công")
        except:
            try:
                # Sử dụng mô hình tiếng Anh nếu không có mô hình tiếng Việt
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                print("Không tìm thấy mô hình tiếng Việt. Sử dụng mô hình tiếng Anh thay thế.")
            except:
                self.nlp = None
                print("Không thể tải mô hình spaCy. Sẽ chỉ sử dụng phương pháp rule-based.")

    def preprocess_text(self, text):
        """Tiền xử lý văn bản"""
        if not text or not isinstance(text, str):
            return ""

        # Chuyển thành chữ thường
        text = text.lower()

        # Loại bỏ các ký tự đặc biệt
        text = re.sub(r'[^\w\s\-\+]', ' ', text)

        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def rule_based_extraction(self, text):
        """Trích xuất kỹ năng dựa trên quy tắc"""
        text = self.preprocess_text(text)
        skills = []

        # Tìm kiếm kỹ năng trong từ điển
        for skill in self.skill_dictionary:
            if re.search(r'\b' + re.escape(skill) + r'\b', text):
                skills.append(skill)

        # Tìm kiếm theo mẫu
        for pattern in self.skill_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                skill_text = match.group(1).strip()
                if skill_text:
                    skills.append(skill_text)

        # Tìm kiếm câu chứa từ khóa chỉ thị
        sentences = re.split(r'[.,;!?]', text)
        for sentence in sentences:
            for indicator in self.skill_indicators:
                if indicator in sentence:
                    # Lấy cụm từ sau từ khóa chỉ thị
                    parts = sentence.split(indicator)
                    if len(parts) > 1:
                        potential_skill = parts[1].strip()
                        if potential_skill and len(potential_skill.split()) <= 5:  # Giới hạn độ dài kỹ năng
                            skills.append(potential_skill)

        return list(set(skills))

    def nlp_based_extraction(self, text):
        """Trích xuất kỹ năng sử dụng NLP"""
        if self.nlp is None:
            return []

        text = self.preprocess_text(text)
        skills = []

        # Phân tích cú pháp
        doc = self.nlp(text)

        # Trích xuất danh từ và cụm danh từ tiềm năng là kỹ năng
        for chunk in doc.noun_chunks:
            # Loại bỏ stopwords
            filtered_chunk = ' '.join([token.text for token in chunk if token.text not in self.stopwords])
            if filtered_chunk:
                skills.append(filtered_chunk)

        # Trích xuất từ dựa trên POS tagging
        for token in doc:
            if token.pos_ in ["NOUN", "ADJ"] and token.text not in self.stopwords:
                skills.append(token.text)

        return list(set(skills))

    def extract_skills_from_job(self, requirements):
        """Trích xuất kỹ năng từ phần yêu cầu công việc"""
        if not requirements:
            return []

        # Kết hợp các phương pháp trích xuất
        rule_skills = self.rule_based_extraction(requirements)
        nlp_skills = []

        if self.nlp is not None:
            nlp_skills = self.nlp_based_extraction(requirements)

        # Gộp và làm sạch kết quả
        combined_skills = list(set(rule_skills + nlp_skills))
        final_skills = []

        for skill in combined_skills:
            skill = skill.strip()
            # Loại bỏ các kỹ năng quá ngắn hoặc quá dài
            if len(skill) > 2 and len(skill.split()) <= 5:
                final_skills.append(skill)

        return final_skills


# Hàm để test việc trích xuất kỹ năng từ văn bản
def test_skill_extraction():
    extractor = SkillExtractor()

    # Danh sách các yêu cầu công việc để test
    test_requirements = [
        """
        - Ưu tên có kinh nghiệm làm việc trong môi trường Thẩm mỹ/Spa
        - Ưu tiên có kinh nghiệm giảng dạy
        - Ưu tiên có các chứng chỉ/chứng nhận sau:
          + Chứng chỉ về da
          + Chứng chỉ vật lý trị liệu
          + Trung cấp y sĩ
        - Yêu cầu có kiến thức về da
        - Có kỹ năng giảng dạy đào tạo, có khả năng truyền đạt tốt
        - Nhanh nhẹn, nhiệt tình, tận tâm với công việc
        - Tuổi từ 22 trở lên
        - Ngoại hình dễ nhìn, da mặt đẹp
        """,

        """
        - Tốt nghiệp đại học chuyên ngành Công nghệ thông tin, Khoa học máy tính hoặc tương đương
        - Có kinh nghiệm lập trình Python tối thiểu 2 năm
        - Hiểu biết về Machine Learning và các thuật toán AI
        - Sử dụng thành thạo các thư viện như TensorFlow, PyTorch, Scikit-learn
        - Có khả năng làm việc độc lập và làm việc nhóm tốt
        - Tiếng Anh đọc hiểu tài liệu chuyên ngành
        - Ưu tiên ứng viên có kinh nghiệm với NLP hoặc Computer Vision
        """,

        """
        - Tốt nghiệp cao đẳng trở lên các ngành Kinh tế, Quản trị kinh doanh, Marketing
        - Có ít nhất 1 năm kinh nghiệm ở vị trí tương đương
        - Kỹ năng giao tiếp và thuyết trình tốt
        - Sử dụng thành thạo Excel, Word, PowerPoint
        - Có khả năng làm việc dưới áp lực cao
        - Nhanh nhẹn, chủ động trong công việc
        - Ưu tiên ứng viên có kinh nghiệm bán hàng hoặc marketing online
        """
    ]

    # Test trích xuất kỹ năng cho mỗi yêu cầu
    for i, requirements in enumerate(test_requirements):
        print(f"\n--- Yêu cầu công việc {i + 1} ---")
        print(requirements.strip())

        # Trích xuất kỹ năng sử dụng rule-based
        rule_skills = extractor.rule_based_extraction(requirements)
        print("\nKỹ năng (Rule-based):")
        for skill in rule_skills:
            print(f"  - {skill}")

        # Trích xuất kỹ năng sử dụng NLP (nếu có)
        if extractor.nlp is not None:
            nlp_skills = extractor.nlp_based_extraction(requirements)
            print("\nKỹ năng (NLP-based):")
            for skill in nlp_skills:
                print(f"  - {skill}")

        # Kết hợp cả hai phương pháp
        combined_skills = extractor.extract_skills_from_job(requirements)
        print("\nKỹ năng (Kết hợp):")
        for skill in combined_skills:
            print(f"  - {skill}")

        print("\n" + "=" * 50)


if __name__ == "__main__":
    test_skill_extraction()