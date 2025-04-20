import PyPDF2
from docx import Document
from PIL import Image
import pytesseract

class FileTextExtractor:
    def __init__(self, allowed_extensions=None):
        """
        初始化文件文本提取器，可自定义允许的文件扩展名。
        
        :param allowed_extensions: 允许的文件扩展名集合，默认为 {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}
        """
        if allowed_extensions is None:
            self.allowed_extensions = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}
        else:
            self.allowed_extensions = allowed_extensions

    def allowed_file(self, filename):
        """
        检查文件名是否具有允许的扩展名。
        
        :param filename: 文件名
        :return: 如果允许返回 True，否则返回 False
        """
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def extract_text(self, filepath, filename):
        """
        根据文件类型提取文本内容。
        
        :param filepath: 文件路径
        :param filename: 文件名（用于确定扩展名）
        :return: 提取的文本内容或错误信息字符串
        """
        ext = filename.rsplit('.', 1)[1].lower()
        text = ""
        try:
            if ext == 'txt':
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
            elif ext == 'pdf':
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                        
            elif ext in ('doc', 'docx'):
                doc = Document(filepath)
                for para in doc.paragraphs:
                    text += para.text + "\n"
                    
            elif ext in ('png', 'jpg', 'jpeg', 'gif'):
                img = Image.open(filepath)
                text = pytesseract.image_to_string(img, lang='chi_sim+eng')
                
        except Exception as e:
            print(f"Error extracting text from {filename}: {str(e)}")
            return f"无法解析文件 {filename} (错误: {str(e)})"
        
        return text.strip()