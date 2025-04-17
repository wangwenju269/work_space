# backend.py (Flask)
from flask import Flask, request, jsonify
import json
from openai import OpenAI
from docling.document_converter import DocumentConverter

app = Flask(__name__)
converter = DocumentConverter()

@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

def extract_content(url):
    try:
        result = converter.convert(url)
        result = result.document.export_to_markdown()
        return result
    except:
        return ""

def generate_with_gpt(prompt):
    client = OpenAI(
        api_key="<your_api_key>",
        base_url="your_base_url>"
    )
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def infer(articles_content, data):
    context = f"""
        请根据以下新闻素材，帮我提炼出以下关键信息，以便我撰写软文：
        1. **核心主题/事件:** 简要概括新闻的核心内容或主要事件。
        2. **可用于软文的亮点/角度:**  提炼出可用于软文写作的亮点、独特角度或切入点。
        [新闻素材]: {articles_content}
    """
    
    refine_context = generate_with_gpt(context)
    prompt = f"""
    请根据以下材料生成3个软文标题及其对应的视角：
    [广告内容]: {data['adContent']}
    [要求]:
    1. 风格：{data['style']}
    2. 突出广告核心优势
    3. 结合最新热点
    4. 口语化吸引眼球
    [新闻素材]: {refine_context}
    输出一个遵循以下格式的JSON列表:
    ```json
    [
        {{
            "title": "软文标题",
            "perspective": "软文新视角"
        }},
        ...
    ]
    ```
    """
    generated = generate_with_gpt(prompt)
    return json.loads(generated)

@app.route('/generate/titles', methods=['POST'])
def generate_titles():
    data = request.json
    articles_content = []
    for url in data['urls']:
        content = extract_content(url.strip())
        articles_content.append(content[:2000])
    articles_content = " ".join(articles_content)    
    title_perspective = infer(articles_content, data)
    return jsonify({"titles_perspectives": title_perspective})

@app.route('/generate/articles', methods=['POST'])
def generate_articles():
    try:
        data = request.json
        if not data or 'selectedTitle' not in data:
            return jsonify({"error": "缺少必要参数"}), 400

        articles_content = []
        for url in data.get('urls', []):
            try:
                content = extract_content(url.strip())
                articles_content.append(content[:2000])
            except Exception as e:
                print(f"URL {url} 内容提取失败: {str(e)}")
                continue

        prompt = f"""
        根据以下素材生成软文：
        [新闻素材]: {" ".join(articles_content)}
        [广告内容]: {data.get('adContent', '')}
        [风格要求]: {data.get('style', '新闻')}
        [选定标题]: {data.get('selectedTitle', '')}
        [选定视角]: {data.get('selectedPerspective', '')}
        
        要求：
        1. 在保留新闻素材的基础上，过渡到广告内容
        2. 突出广告核心优势
        3. 结构清晰，段落分明
        4. 输出文章,markdown格式,长度约1000字
        """
        
        generated_article = generate_with_gpt(prompt)
        return jsonify({"article": generated_article})

    except Exception as e:
        print(f"文章生成异常: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=7568)