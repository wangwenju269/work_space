from flask import Flask, request, jsonify, Response
import time
import os
import json
from werkzeug.utils import secure_filename
import PyPDF2
from docx import Document
from PIL import Image
import pytesseract
from search import ResearchWorkflow
lab = ResearchWorkflow()
app = Flask(__name__)

# 配置文件上传
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(filepath, filename):
    """根据文件类型提取文本内容"""
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
            # 使用OCR提取图片中的文字
            img = Image.open(filepath)
            text = pytesseract.image_to_string(img, lang='chi_sim+eng')
            
    except Exception as e:
        print(f"Error extracting text from {filename}: {str(e)}")
        return f"无法解析文件 {filename} (错误: {str(e)})"
    
    return text.strip()


@app.route('/')
def index():
    # 返回前端HTML页面
    with open('templates/index.html', 'r', encoding='utf-8') as f:
        return f.read()


# 新增一个专门的文件上传和解析接口
@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': '没有文件上传'}), 400
    
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({'success': False, 'error': '没有选择文件'}), 400
    
    filenames = []
    file_contents = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename) or f'file_{int(time.time())}'
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                filenames.append(file.filename)
                
                # 解析文件内容（即使不在前端显示，后端仍然需要解析）
                combined_text = "上传的文件内容：\n\n"
                content = extract_text_from_file(filepath, file.filename)
                combined_text += f"=== 文件: {file.filename} ===\n"
                combined_text += content + "\n\n"
                file_contents.append({
                    'filename': file.filename,
                    'content': combined_text
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'文件 {file.filename} 解析失败: {str(e)}'
                }), 400
    
    if not filenames:
        return jsonify({'success': False, 'error': '没有有效的文件被上传'}), 400
    
    return jsonify({
        'success': True,
        'filenames': filenames,
        'contents': file_contents  # 虽然前端不显示，但后端仍然返回内容供后续处理
    })


# 修改原有的聊天接口，不再处理文件上传
@app.route('/api/chat', methods=['POST'])  # 确保只允许POST方法
def chat():
    # 获取JSON格式的请求体
    if not request.is_json:
        return jsonify({'error': '请求必须是JSON格式'}), 400
    
    data = request.get_json()
    user_message = data.get('message', '')
    use_web_search = data.get('web_search', False)  #待实现
    file_contents = data.get('contents', [])        #待实现  
    
    lab.research_topic = user_message
    # 验证请求数据
    if not user_message and not file_contents:
        return jsonify({'error': '消息或文件不能为空'}), 400
    
    # 流式响应生成器
    def generate():
        
        for chunk in lab.literature_review() :
            yield f"data: {json.dumps({'message': chunk},ensure_ascii=False)}\n\n"
            
        yield f"data: {json.dumps({'message': '已完成文献检索工作'},ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=7862)