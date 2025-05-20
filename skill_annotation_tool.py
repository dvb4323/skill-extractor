import json
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re


class SkillAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Công cụ gán nhãn kỹ năng")
        self.root.geometry("1200x800")

        # Dữ liệu
        self.jobs = []
        self.current_job_idx = 0
        self.annotations = {}
        self.selected_skill = None
        self.skill_spans = []

        # Tạo giao diện
        self.create_widgets()

        # Tải dữ liệu gán nhãn nếu có
        self.annotations_file = "data/annotations.json"
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)

    def create_widgets(self):
        # Frame chính
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Khung thông tin job
        job_info_frame = ttk.LabelFrame(main_frame, text="Thông tin công việc", padding=10)
        job_info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.job_title_var = tk.StringVar()
        ttk.Label(job_info_frame, text="Tiêu đề:").grid(row=0, column=0, sticky="w")
        ttk.Label(job_info_frame, textvariable=self.job_title_var, font=("Arial", 10, "bold")).grid(row=0, column=1,
                                                                                                    sticky="w")

        self.company_var = tk.StringVar()
        ttk.Label(job_info_frame, text="Công ty:").grid(row=1, column=0, sticky="w")
        ttk.Label(job_info_frame, textvariable=self.company_var).grid(row=1, column=1, sticky="w")

        # Khung yêu cầu công việc
        requirements_frame = ttk.LabelFrame(main_frame, text="Yêu cầu công việc", padding=10)
        requirements_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.requirements_text = tk.Text(requirements_frame, wrap=tk.WORD, font=("Arial", 11))
        self.requirements_text.pack(fill=tk.BOTH, expand=True)
        self.requirements_text.bind("<ButtonRelease-1>", self.on_text_select)

        # Khung kỹ năng đã gán nhãn
        skills_frame = ttk.LabelFrame(main_frame, text="Kỹ năng đã gán nhãn", padding=10)
        skills_frame.pack(fill=tk.X, padx=5, pady=5)

        self.skills_treeview = ttk.Treeview(skills_frame, columns=("skill", "start", "end"), show="headings")
        self.skills_treeview.heading("skill", text="Kỹ năng")
        self.skills_treeview.heading("start", text="Vị trí bắt đầu")
        self.skills_treeview.heading("end", text="Vị trí kết thúc")
        self.skills_treeview.column("skill", width=300)
        self.skills_treeview.column("start", width=100)
        self.skills_treeview.column("end", width=100)
        self.skills_treeview.pack(fill=tk.X, expand=True)
        self.skills_treeview.bind("<<TreeviewSelect>>", self.on_skill_select)

        # Khung điều khiển
        control_frame = ttk.Frame(main_frame, padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(control_frame, text="Thêm kỹ năng", command=self.add_skill).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Xóa kỹ năng", command=self.delete_skill).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Job trước", command=self.prev_job).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Job tiếp theo", command=self.next_job).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Lưu gán nhãn", command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Tải dữ liệu", command=self.load_data).pack(side=tk.LEFT, padx=5)

        # Thanh trạng thái
        self.status_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.status_var, font=("Arial", 10, "italic")).pack(fill=tk.X, padx=5,
                                                                                               pady=5)

    def load_data(self):
        """Tải dữ liệu job từ file"""
        file_path = filedialog.askopenfilename(
            title="Chọn file dữ liệu job",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.jobs = json.load(f)

            self.current_job_idx = 0
            self.display_current_job()
            self.status_var.set(f"Đã tải {len(self.jobs)} jobs từ {file_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải dữ liệu: {e}")

    def display_current_job(self):
        """Hiển thị thông tin job hiện tại"""
        if not self.jobs or self.current_job_idx >= len(self.jobs):
            self.status_var.set("Không có dữ liệu job")
            return

        job = self.jobs[self.current_job_idx]
        job_id = str(job.get("id", hash(job.get("title", ""))))

        # Hiển thị thông tin job
        self.job_title_var.set(job.get("title", ""))
        self.company_var.set(job.get("company_name", ""))

        # Hiển thị requirements
        requirements = job.get("requirements", "")
        self.requirements_text.delete(1.0, tk.END)
        self.requirements_text.insert(tk.END, requirements)

        # Hiển thị các kỹ năng đã gán nhãn
        self.skills_treeview.delete(*self.skills_treeview.get_children())
        self.skill_spans = []

        if job_id in self.annotations:
            self.skill_spans = self.annotations[job_id].get("skills", [])
            for i, span in enumerate(self.skill_spans):
                self.skills_treeview.insert("", tk.END, iid=str(i), values=(
                    span["skill"],
                    span["start_word"],
                    span["end_word"]
                ))

        self.status_var.set(f"Job {self.current_job_idx + 1}/{len(self.jobs)}")

    def on_text_select(self, event):
        """Xử lý sự kiện khi người dùng chọn văn bản"""
        try:
            # Lấy văn bản đã chọn
            if self.requirements_text.tag_ranges(tk.SEL):
                start = self.requirements_text.index(tk.SEL_FIRST)
                end = self.requirements_text.index(tk.SEL_LAST)
                selected_text = self.requirements_text.get(start, end)

                if selected_text:
                    # Tính toán vị trí từ
                    full_text = self.requirements_text.get(1.0, tk.END)
                    words = full_text.split()

                    # Tìm vị trí bắt đầu và kết thúc của văn bản đã chọn
                    start_word_idx = len(full_text[:self.requirements_text.index(tk.SEL_FIRST)].split())
                    end_word_idx = start_word_idx + len(selected_text.split()) - 1

                    # Hiển thị dialog xác nhận
                    if messagebox.askyesno("Thêm kỹ năng", f"Thêm '{selected_text}' là kỹ năng?"):
                        self.add_skill_span(selected_text, start_word_idx, end_word_idx)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi chọn văn bản: {e}")

    def add_skill_span(self, skill, start_word, end_word):
        """Thêm một kỹ năng vào danh sách"""
        skill_span = {
            "skill": skill,
            "start_word": start_word,
            "end_word": end_word
        }

        self.skill_spans.append(skill_span)
        self.skills_treeview.insert("", tk.END, iid=str(len(self.skill_spans) - 1), values=(
            skill,
            start_word,
            end_word
        ))

        # Lưu vào annotations
        job = self.jobs[self.current_job_idx]
        job_id = str(job.get("id", hash(job.get("title", ""))))

        if job_id not in self.annotations:
            self.annotations[job_id] = {
                "requirements": job.get("requirements", ""),
                "skills": []
            }

        self.annotations[job_id]["skills"] = self.skill_spans

    def on_skill_select(self, event):
        """Xử lý sự kiện khi người dùng chọn một kỹ năng trong danh sách"""
        selected_items = self.skills_treeview.selection()
        if not selected_items:
            self.selected_skill = None
            return

        self.selected_skill = int(selected_items[0])

    def add_skill(self):
        """Thêm kỹ năng thủ công"""
        selected_text = ""
        if self.requirements_text.tag_ranges(tk.SEL):
            selected_text = self.requirements_text.get(tk.SEL_FIRST, tk.SEL_LAST)

        # Hiển thị dialog nhập kỹ năng
        skill = tk.simpledialog.askstring("Thêm kỹ năng", "Nhập kỹ năng:", initialvalue=selected_text)
        if not skill:
            return

        # Tìm kỹ năng trong văn bản
        requirements = self.requirements_text.get(1.0, tk.END)
        words = requirements.split()
        skill_words = skill.split()

        # Tìm vị trí đầu tiên của kỹ năng trong văn bản
        for i in range(len(words) - len(skill_words) + 1):
            match = True
            for j in range(len(skill_words)):
                if i + j >= len(words) or words[i + j].lower() != skill_words[j].lower():
                    match = False
                    break

            if match:
                self.add_skill_span(skill, i, i + len(skill_words) - 1)
                return

        # Nếu không tìm thấy, hỏi người dùng có muốn thêm thủ công không
        if messagebox.askyesno("Thêm kỹ năng",
                               "Không tìm thấy kỹ năng trong văn bản. Bạn có muốn thêm thủ công không?"):
            start_word = tk.simpledialog.askinteger("Vị trí bắt đầu", "Nhập vị trí từ bắt đầu:", initialvalue=0)
            end_word = tk.simpledialog.askinteger("Vị trí kết thúc", "Nhập vị trí từ kết thúc:",
                                                  initialvalue=start_word + len(skill_words) - 1)

            if start_word is not None and end_word is not None:
                self.add_skill_span(skill, start_word, end_word)

    def delete_skill(self):
        """Xóa kỹ năng đã chọn"""
        if self.selected_skill is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một kỹ năng để xóa")
            return

        # Xóa khỏi danh sách hiển thị
        self.skills_treeview.delete(str(self.selected_skill))

        # Xóa khỏi danh sách kỹ năng
        self.skill_spans.pop(self.selected_skill)

        # Cập nhật lại ID của các item trong treeview
        self.skills_treeview.delete(*self.skills_treeview.get_children())
        for i, span in enumerate(self.skill_spans):
            self.skills_treeview.insert("", tk.END, iid=str(i), values=(
                span["skill"],
                span["start_word"],
                span["end_word"]
            ))

        # Cập nhật lại annotations
        job = self.jobs[self.current_job_idx]
        job_id = str(job.get("id", hash(job.get("title", ""))))

        if job_id in self.annotations:
            self.annotations[job_id]["skills"] = self.skill_spans

        self.selected_skill = None

    def prev_job(self):
        """Chuyển đến job trước đó"""
        if not self.jobs:
            return

        self.current_job_idx = (self.current_job_idx - 1) % len(self.jobs)
        self.display_current_job()

    def next_job(self):
        """Chuyển đến job tiếp theo"""
        if not self.jobs:
            return

        self.current_job_idx = (self.current_job_idx + 1) % len(self.jobs)
        self.display_current_job()

    def save_annotations(self):
        """Lưu dữ liệu gán nhãn ra file"""
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(self.annotations_file), exist_ok=True)

        with open(self.annotations_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=4)

        messagebox.showinfo("Thông báo", f"Đã lưu dữ liệu gán nhãn vào {self.annotations_file}")

        # Tạo dữ liệu huấn luyện dưới dạng BIO
        self.create_bio_training_data()

    def create_bio_training_data(self):
        """Tạo dữ liệu huấn luyện dưới dạng BIO (Beginning, Inside, Outside)"""
        bio_data = []

        for job_id, annotation in self.annotations.items():
            requirements = annotation.get("requirements", "")
            skills = annotation.get("skills", [])

            if not requirements or not skills:
                continue

            # Tách văn bản thành các từ
            words = requirements.split()

            # Tạo list nhãn ban đầu, tất cả là "O" (Outside)
            labels = ["O"] * len(words)

            # Gán nhãn cho các kỹ năng
            for skill in skills:
                start_word = skill.get("start_word", 0)
                end_word = skill.get("end_word", 0)

                if start_word >= len(labels):
                    continue

                # Gán nhãn "B-SKILL" cho từ đầu tiên của kỹ năng
                labels[start_word] = "B-SKILL"

                # Gán nhãn "I-SKILL" cho các từ còn lại của kỹ năng
                for i in range(start_word + 1, min(end_word + 1, len(labels))):
                    labels[i] = "I-SKILL"

            # Thêm vào dữ liệu huấn luyện
            bio_data.append({
                "job_id": job_id,
                "words": words,
                "labels": labels
            })

        # Lưu dữ liệu BIO ra file
        with open("data/bio_training_data.json", 'w', encoding='utf-8') as f:
            json.dump(bio_data, f, ensure_ascii=False, indent=4)

        # Tạo file CSV cho huấn luyện
        self.create_csv_training_data(bio_data)

    def create_csv_training_data(self, bio_data):
        """Tạo file CSV cho huấn luyện từ dữ liệu BIO"""
        csv_rows = []

        for data in bio_data:
            words = data["words"]
            labels = data["labels"]

            for word, label in zip(words, labels):
                csv_rows.append({
                    "word": word,
                    "label": label,
                    "job_id": data["job_id"]
                })

            # Thêm dòng trống để phân tách các câu
            csv_rows.append({
                "word": "",
                "label": "",
                "job_id": data["job_id"]
            })

        # Tạo DataFrame và lưu ra file CSV
        df = pd.DataFrame(csv_rows)
        df.to_csv("data/bio_training_data.csv", index=False, encoding='utf-8')

        messagebox.showinfo("Thông báo",
                            "Đã tạo dữ liệu huấn luyện BIO tại data/bio_training_data.json và data/bio_training_data.csv")


def main():
    root = tk.Tk()
    app = SkillAnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()