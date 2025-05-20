import pandas as pd
import json
import os
import nltk
from tqdm import tqdm
from skill_extraction_system import SkillExtractor  # Import từ file đã tạo trước đó

# Đảm bảo đã tải các tài nguyên cần thiết
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def load_job_data(file_path):
    """Tải dữ liệu job từ file JSON"""
    if not os.path.exists(file_path):
        print(f"File không tồn tại: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            if file_path.endswith('.json'):
                return json.load(f)
            else:
                print(f"Định dạng file không được hỗ trợ: {file_path}")
                return []
        except json.JSONDecodeError:
            print(f"Lỗi đọc file JSON: {file_path}")
            return []


def process_excel_job_data(file_path):
    """Xử lý dữ liệu job từ file Excel"""
    try:
        df = pd.read_excel(file_path)
        # Điều chỉnh tên cột nếu cần
        column_mapping = {
            'job_title': 'title',
            'job_description': 'job_description',
            'job_requirements': 'requirements',
            # Thêm các ánh xạ cột khác nếu cần
        }

        # Đổi tên cột nếu cần
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)

        # Chuyển DataFrame thành danh sách các dictionary
        return df.to_dict(orient='records')
    except Exception as e:
        print(f"Lỗi xử lý file Excel {file_path}: {e}")
        return []


def main():
    # Thư mục chứa dữ liệu job
    data_dir = "../data"
    output_dir = "../output"

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Khởi tạo bộ trích xuất kỹ năng
    extractor = SkillExtractor()

    # Tìm tất cả các file dữ liệu
    all_jobs = []

    # Xử lý các file JSON
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    for file in tqdm(json_files, desc="Đang xử lý file JSON"):
        file_path = os.path.join(data_dir, file)
        jobs = load_job_data(file_path)
        all_jobs.extend(jobs)

    # Xử lý các file Excel
    excel_files = [f for f in os.listdir(data_dir) if f.endswith(('.xlsx', '.xls'))]
    for file in tqdm(excel_files, desc="Đang xử lý file Excel"):
        file_path = os.path.join(data_dir, file)
        jobs = process_excel_job_data(file_path)
        all_jobs.extend(jobs)

    print(f"Tổng số job đã tải: {len(all_jobs)}")

    # Trích xuất kỹ năng từ tất cả các job
    print("Đang tiến hành trích xuất kỹ năng...")
    analysis = extractor.analyze_job_batch(all_jobs)

    # Lưu kết quả
    output_file = os.path.join(output_dir, "job_skills_analysis.json")
    extractor.save_results(analysis, output_file)

    # Tạo bộ dữ liệu đã được làm giàu với các kỹ năng đã trích xuất
    enriched_jobs = []
    for i, job in enumerate(all_jobs):
        job_copy = job.copy()
        job_copy["extracted_skills"] = analysis["job_analyses"][i]["extracted_skills"] if i < len(
            analysis["job_analyses"]) else []
        enriched_jobs.append(job_copy)

    # Lưu bộ dữ liệu đã được làm giàu
    enriched_output = os.path.join(output_dir, "enriched_jobs.json")
    with open(enriched_output, 'w', encoding='utf-8') as f:
        json.dump(enriched_jobs, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu dữ liệu job đã được làm giàu vào {enriched_output}")

    # Tạo file báo cáo thống kê
    generate_skill_statistics(analysis, output_dir)


def generate_skill_statistics(analysis, output_dir):
    """Tạo báo cáo thống kê về kỹ năng"""
    # Thống kê kỹ năng phổ biến
    with open(os.path.join(output_dir, "skill_statistics.txt"), 'w', encoding='utf-8') as f:
        f.write("BÁO CÁO THỐNG KÊ KỸ NĂNG\n")
        f.write("=" * 50 + "\n\n")

        f.write("Top 20 kỹ năng phổ biến nhất:\n")
        for i, (skill, count) in enumerate(analysis["most_common_skills"], 1):
            f.write(f"{i}. {skill}: {count} lần xuất hiện\n")

        f.write("\n" + "=" * 50 + "\n\n")

        f.write("Phân cụm kỹ năng:\n")
        for cluster_name, terms in analysis["skill_clusters"].items():
            f.write(f"\n{cluster_name}:\n")
            for term in terms:
                f.write(f"- {term}\n")

    print(f"Đã tạo báo cáo thống kê tại {os.path.join(output_dir, 'skill_statistics.txt')}")


if __name__ == "__main__":
    main()