from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS  # 添加CORS支持
import time
import os
import json
import logging
from werkzeug.utils import secure_filename
from uuid import uuid4
from search import ResearchWorkflow, AcademicWorkflow
from utils import FileTextExtractor

# 初始化基础组件
extractor = FileTextExtractor()

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用CORS支持

# 配置参数
app.config.update({
    'UPLOAD_FOLDER': 'uploads',
    'MAX_CONTENT_LENGTH': 100 * 1024 * 1024,  # 100MB限制
    'ALLOWED_EXTENSIONS': set(['pdf', 'docx', 'txt', 'md']),
    'SESSION_FILES': {}  # 临时存储会话文件
})

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = app.logger

# 辅助函数
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_session_id():
    return str(uuid4())

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/api/upload', methods=['POST'])
def handle_upload():
    """处理文件上传并创建会话"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No selected files'}), 400

    session_id = generate_session_id()
    session_files = []
    
    try:
        for file in files:
            if file and allowed_file(file.filename):
                # 生成安全文件名
                original_name = secure_filename(file.filename)
                file_id = f"{int(time.time())}_{original_name}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
                
                # 保存文件
                file.save(file_path)
                
                # 提取文本内容
                content = extractor.extract_text(file_path)
                session_files.append({
                    'original_name': original_name,
                    'file_path': file_path,
                    'content': content
                })
        
        # 创建会话记录
        app.config['SESSION_FILES'][session_id] = {
            'files': session_files,
            'created_at': time.time()
        }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'filenames': [f['original_name'] for f in session_files]
        })
    
    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")
        return jsonify({'error': 'File processing failed'}), 500

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """处理聊天请求"""
    data = request.get_json()
    user_message = data['message']
    use_web_search = data.get('web_search', False)
    use_axiver_search = data.get('axiver_search', False)
    if use_web_search:
        search_workflow = ResearchWorkflow(max_steps=15)
    elif use_axiver_search:     
        search_workflow = AcademicWorkflow(max_steps=15)
    
    def _format_message(content):
        """统一格式化消息为SSE格式"""
        return f"data: {json.dumps({'message': content}, ensure_ascii=False)}\n\n"  
       
    def generate_stream():
            try:
                for chunk in search_workflow.run(research_topic=user_message):
                    yield _format_message(chunk)
                
                # 生成最终分析报告
                report_content = search_workflow.generate_llm_report()
                yield _format_message({
                        'type': 'FINAL_REPORT',
                        'content': report_content
                })
                
            except Exception as e:
                logger.error(f"Stream generation failed: {str(e)}")
                yield _format_message({
                        'type': 'ERROR',
                        'content': f'处理请求时发生错误：{str(e)}'
                }) 
                
            yield _format_message({
                        'type': 'TASK_COMPLETE',
                        'content': '✅ 任务已完成'
                })    
           
    return Response(
        generate_stream(),
        mimetype='text/event-stream'
    )

@app.route('/uploads/<filename>')
def serve_file(filename):
    """文件访问端点"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # 确保上传目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=7862, debug=True)