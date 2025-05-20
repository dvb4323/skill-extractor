# skill_dictionaries.py

# Danh sách kỹ năng mềm
SOFT_SKILLS = [
    "giao tiếp", "thuyết trình", "thuyết phục", "đàm phán", "thương lượng",
    "làm việc nhóm", "teamwork", "hợp tác", "lãnh đạo", "quản lý",
    "tư duy phản biện", "critical thinking", "giải quyết vấn đề", "problem solving",
    "sáng tạo", "creativity", "đổi mới", "innovation", "sáng kiến", "initiative",
    "quản lý thời gian", "time management", "tổ chức", "organization",
    "chịu áp lực", "stress management", "làm việc dưới áp lực", "kiểm soát căng thẳng",
    "thích nghi", "adaptation", "linh hoạt", "flexibility", "khả năng thích ứng",
    "tự học", "self-learning", "học hỏi", "cầu tiến", "tự phát triển",
    "trung thực", "honesty", "chính trực", "integrity", "đạo đức",
    "nhiệt tình", "enthusiasm", "năng động", "dynamic", "chủ động", "proactive",
    "tỉ mỉ", "detail-oriented", "cẩn thận", "careful", "tỉ mỉ", "chi tiết",
    "quyết đoán", "decisive", "ra quyết định", "decision making",
    "đồng cảm", "empathy", "thấu hiểu", "understanding", "chia sẻ",
    "tự tin", "confidence", "tự tin", "self-confidence",
    "khả năng đọc hiểu", "reading comprehension", "đọc hiểu",
    "tư duy logic", "logical thinking", "tư duy hệ thống", "systematic thinking",
    "tư duy chiến lược", "strategic thinking", "chiến lược",
    "phân tích", "analytical", "phân tích", "analysis", "khả năng phân tích",
    "kỹ năng giao tiếp", "communication skills", "kỹ năng viết", "writing skills",
    "kỹ năng lắng nghe", "listening skills", "lắng nghe", "listening",
    "khả năng học hỏi", "learning ability", "học hỏi nhanh", "nhanh nhẹn",
    "kiên nhẫn", "patience", "kiên nhẫn", "kiên trì",
    "tích cực", "positive", "tư duy tích cực", "positive thinking",
    "hòa đồng", "sociable", "hòa đồng", "thân thiện", "friendly",
]

# Danh sách kỹ năng chuyên môn
HARD_SKILLS = [
    # IT & Programming
    "python", "java", "javascript", "c++", "c#", "php", "html", "css", "sql",
    "node.js", "react", "angular", "vue.js", "django", "flask", "spring", "hibernate",
    "docker", "kubernetes", "git", "github", "gitlab", "aws", "azure", "gcp",
    "machine learning", "deep learning", "ai", "nlp", "computer vision", "data mining",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "scipy",
    "data analysis", "data science", "big data", "hadoop", "spark", "tableau", "power bi",
    "database", "mysql", "postgresql", "mongodb", "oracle", "sql server", "nosql",
    "restful api", "graphql", "microservices", "devops", "ci/cd", "jenkins", "travis ci",
    "agile", "scrum", "kanban", "waterfall", "jira", "confluence", "trello",
    "mobile development", "android", "ios", "react native", "flutter", "swift", "kotlin",
    "wordpress", "seo", "sem", "google analytics", "google ads", "facebook ads",

    # Business & Finance
    "kế toán", "accounting", "tài chính", "finance", "ngân hàng", "banking",
    "excel", "word", "powerpoint", "ms office", "văn phòng", "office",
    "erp", "crm", "sap", "oracle financials", "quickbooks", "myob", "misa", "fast",
    "báo cáo tài chính", "financial reporting", "thuế", "tax", "kiểm toán", "audit",
    "phân tích tài chính", "financial analysis", "đầu tư", "investment",
    "ngân sách", "budgeting", "dự báo", "forecasting", "kế toán quản trị", "management accounting",
    "bảo hiểm", "insurance", "chứng khoán", "securities", "forex", "trading",

    # Languages
    "tiếng anh", "english", "tiếng nhật", "japanese", "tiếng trung", "chinese",
    "tiếng hàn", "korean", "tiếng pháp", "french", "tiếng đức", "german",
    "toeic", "ielts", "toefl", "n1", "n2", "n3", "n4", "n5", "hsk", "topik",

    # Engineering & Technical
    "autocad", "solidworks", "3d modeling", "revit", "sketchup", "catia", "ansys",
    "thiết kế", "design", "bản vẽ", "drawing", "quy hoạch", "planning",
    "civil 3d", "gis", "surveying", "đo đạc", "mapping", "bản đồ",
    "plc", "scada", "automation", "tự động hóa", "điều khiển", "control",
    "cơ khí", "mechanical", "điện", "electrical", "điện tử", "electronics",
    "xây dựng", "construction", "kiến trúc", "architecture", "kết cấu", "structure",

    # Marketing & Design
    "marketing", "digital marketing", "content marketing", "inbound marketing",
    "social media", "facebook", "instagram", "tiktok", "youtube", "linkedin",
    "photoshop", "illustrator", "indesign", "premiere pro", "after effects", "lightroom",
    "ui/ux", "user interface", "user experience", "web design", "graphic design",
    "branding", "thương hiệu", "pr", "public relations", "quảng cáo", "advertising",
    "copywriting", "content writing", "content creation", "sáng tạo nội dung",

    # Healthcare
    "y tế", "healthcare", "dược", "pharmacy", "điều dưỡng", "nursing",
    "phẫu thuật", "surgery", "chẩn đoán", "diagnosis", "điều trị", "treatment",
    "chăm sóc bệnh nhân", "patient care", "y học", "medicine", "sức khỏe", "health",

    # Logistics & Supply Chain
    "logistic", "vận tải", "transportation", "kho vận", "warehouse", "xuất nhập khẩu", "import/export",
    "supply chain", "chuỗi cung ứng", "inventory", "tồn kho", "sourcing", "procurement", "mua hàng",
    "quản lý dự án", "project management", "quản lý chuỗi cung ứng", "supply chain management",

    # Legal
    "luật", "law", "pháp lý", "legal", "hợp đồng", "contract", "tuân thủ", "compliance",
    "sở hữu trí tuệ", "intellectual property", "giải quyết tranh chấp", "dispute resolution",

    # Specialized Industry Terms
    "bất động sản", "real estate", "môi giới", "broker", "định giá", "valuation",
    "nhà hàng", "restaurant", "khách sạn", "hotel", "du lịch", "tourism", "lữ hành", "travel",
    "bán lẻ", "retail", "bán hàng", "sales", "chăm sóc khách hàng", "customer service",
    "giáo dục", "education", "dạy học", "teaching", "đào tạo", "training",
    "nông nghiệp", "agriculture", "thủy sản", "aquaculture", "lâm nghiệp", "forestry",
]