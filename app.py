import streamlit as st
from skill_extractor import SkillExtractor


# Khởi tạo extractor
@st.cache_resource
def load_extractor():
    return SkillExtractor(use_spacy=True)


extractor = load_extractor()

# Tạo giao diện
st.title("Hệ thống trích xuất kỹ năng từ yêu cầu công việc")

# Nhập văn bản
text_input = st.text_area("Nhập yêu cầu công việc:", height=200)

# Nút trích xuất
if st.button("Trích xuất kỹ năng"):
    if text_input:
        with st.spinner("Đang trích xuất kỹ năng..."):
            # Trích xuất kỹ năng
            skills = extractor.extract_skills(text_input)

            # Hiển thị kết quả
            st.subheader("Kỹ năng mềm:")
            if skills['soft_skills']:
                for skill in skills['soft_skills']:
                    st.write(f"- {skill}")
            else:
                st.write("Không tìm thấy kỹ năng mềm.")

            st.subheader("Kỹ năng chuyên môn:")
            if skills['hard_skills']:
                for skill in skills['hard_skills']:
                    st.write(f"- {skill}")
            else:
                st.write("Không tìm thấy kỹ năng chuyên môn.")

            # Hiển thị tổng quan
            st.info(
                f"Đã tìm thấy {len(skills['soft_skills'])} kỹ năng mềm và {len(skills['hard_skills'])} kỹ năng chuyên môn.")
    else:
        st.error("Vui lòng nhập yêu cầu công việc!")

# Hiển thị thông tin
st.sidebar.header("Thông tin")
st.sidebar.markdown("""
Hệ thống này trích xuất kỹ năng từ yêu cầu công việc và phân loại thành kỹ năng mềm và kỹ năng chuyên môn.

**Kỹ năng mềm** là các kỹ năng liên quan đến giao tiếp, làm việc nhóm, tư duy, thái độ...

**Kỹ năng chuyên môn** là các kỹ năng kỹ thuật, công nghệ, ngôn ngữ, công cụ...
""")

# Chạy với lệnh: streamlit run app.py