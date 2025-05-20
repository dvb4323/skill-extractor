import pandas as pd
import re
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Đảm bảo đã tải các tài nguyên cần thiết
nltk.download('punkt')
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
            self.nlp = spacy.load("vi_core_news_lg")
        except:
            # Sử dụng mô hình tiếng Anh nếu không có mô hình tiếng Việt
            print("Không tìm thấy mô hình tiếng Việt. Sử dụng mô hình tiếng Anh thay thế.")
            self.nlp = spacy.load("en_core_web_sm")

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

    def tfidf_clustering(self, texts, n_clusters=10):
        """Phân cụm các kỹ năng sử dụng TF-IDF và KMeans"""
        # Véc-tơ hóa văn bản
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=list(self.stopwords),
            ngram_range=(1, 2)
        )

        # Chuyển đổi văn bản thành ma trận TF-IDF
        X = vectorizer.fit_transform(texts)

        # Phân cụm
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)

        # Lấy các từ quan trọng nhất trong mỗi cụm
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()

        clusters = {}
        for i in range(n_clusters):
            cluster_terms = [terms[ind] for ind in order_centroids[i, :10]]
            clusters[f"cluster_{i}"] = cluster_terms

        return clusters

    def extract_skills_from_job(self, requirements):
        """Trích xuất kỹ năng từ phần yêu cầu công việc"""
        if not requirements:
            return []

        # Kết hợp các phương pháp trích xuất
        rule_skills = self.rule_based_extraction(requirements)
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

    def analyze_job_batch(self, jobs_data):
        """Phân tích một loạt các job và trích xuất kỹ năng"""
        results = []
        all_requirements = []

        for job in jobs_data:
            requirements = job.get("requirements", "")
            all_requirements.append(requirements)

            skills = self.extract_skills_from_job(requirements)

            result = {
                "job_title": job.get("title", ""),
                "company": job.get("company_name", ""),
                "extracted_skills": skills
            }
            results.append(result)

        # Phân tích tần suất kỹ năng
        all_skills = []
        for result in results:
            all_skills.extend(result["extracted_skills"])

        skill_frequency = Counter(all_skills)
        most_common_skills = skill_frequency.most_common(20)

        # Phân cụm kỹ năng từ toàn bộ dữ liệu
        if len(all_requirements) > 10:  # Cần đủ dữ liệu để phân cụm
            skill_clusters = self.tfidf_clustering(all_requirements)
        else:
            skill_clusters = {}

        analysis = {
            "job_analyses": results,
            "most_common_skills": most_common_skills,
            "skill_clusters": skill_clusters
        }

        return analysis

    def save_results(self, analysis, output_file="skill_analysis_results.json"):
        """Lưu kết quả phân tích ra file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu kết quả phân tích vào file {output_file}")


# Ví dụ sử dụng
if __name__ == "__main__":
    # Đọc dữ liệu mẫu
    sample_job = {
        "title": "Nhân Viên Đạo Tạo Về Da Thẩm Mỹ (Yêu Cầu Nữ)",
        "salary": "10 - 15 triệu",
        "location": "Hà Nội & 3 nơi khác",
        "experience": "2 năm",
        "company_name": "Công ty TNHH Mediworld",
        "job_description": "- Đào tạo kiến thức về da thẩm mỹ\n- Đào tạo kiến thức về sản phẩm mỹ phẩm\n- Hỗ trợ đội ngũ kinh doanh triển khai sản phẩm ra thị trường\n- Hỗ trợ các show sự kiện, hội thảo khách hàng, ....",
        "requirements": "- Ưu tên có kinh nghiệm làm việc trong môi trường Thẩm mỹ/Spa\n- Ưu tiên có kinh nghiệm giảng dạy\n- Ưu tiên có các chứng chỉ/chứng nhận sau:\n+ Chứng chỉ về da\n+ Chứng chỉ vật lý trị liệu\n+ Trung cấp y sĩ\n- Yêu cầu có kiến thức về da\n- Có kỹ năng giảng dạy đào tạo, có khả năng truyền đạt tốt\n- Nhanh nhẹn, nhiệt tình, tận tâm với công việc\n- Tuổi từ 22 trở lên\n- Ngoại hình dễ nhìn, da mặt đẹp"
    }

    extractor = SkillExtractor()

    # Trích xuất kỹ năng từ một job
    skills = extractor.extract_skills_from_job(sample_job["requirements"])
    print("Kỹ năng trích xuất:", skills)

    # Phân tích nhiều job
    sample_jobs = [sample_job]  # Thêm nhiều jobs vào đây
    analysis = extractor.analyze_job_batch(sample_jobs)

    # In kết quả
    print("\nKết quả phân tích:")
    print(f"- Số lượng jobs đã phân tích: {len(analysis['job_analyses'])}")
    print("- Các kỹ năng phổ biến nhất:")
    for skill, count in analysis["most_common_skills"]:
        print(f"  * {skill}: {count}")

    # Lưu kết quả
    extractor.save_results(analysis)