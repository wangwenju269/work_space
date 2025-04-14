# Research Assistant Web Application

## 项目简介
基于Flask框架构建的研究辅助系统，集成学术文献检索、多格式文档解析和智能对话功能，支持流式响应处理。

## 核心功能
### 文件解析能力
- 支持格式: PDF/DOCX/PNG/JPG/TXT
- 文本提取方法:
  - PDF: PyPDF2文本提取
  - DOCX: python-docx库解析
  - 图片: pytesseract OCR识别

### 文献研究流程
1. 智能生成检索关键词
2. Arxiv论文摘要检索
3. 全文内容自动下载
4. 文献分析报告生成

### API接口
#### 文件上传接口
```http
POST /api/upload
Content-Type: multipart/form-data

响应格式:
{
    "success": boolean,
    "filenames": [string],
    "contents": [
        {
            "filename": string,
            "content": string
        }
    ]
}
```

#### 研究对话接口
```http
POST /api/chat
Content-Type: application/json

请求体:
{
    "message": "研究主题",
    "web_search": false,
    "contents": ["上传文件内容"]
}

响应格式: text/event-stream
```

### 开发环境
#### 依赖安装
```bash
sudo apt install tesseract-ocr tesseract-ocr-chi-sim
# Python依赖
pip install -r requirements.txt
```

### 系统架构
```bash
.
├── app.py              # Flask主程序
├── search.py           # 研究流程实现
│   ├── ArxivSearch     # 论文检索模块
│   └── ResearchWorkflow # 工作流引擎
├── uploads/            # 文件存储目录
└── templates/
    └── index.html      # 前端界面
```bash    
