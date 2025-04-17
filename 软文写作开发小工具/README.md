# AI 软文生成工具

## 项目介绍

这是一个基于 Flask 和 OpenAI API 的 AI 软文生成工具。用户可以通过输入新闻素材 URL 和广告内容，选择写作风格，生成多个软文标题及其对应的视角，并最终生成一篇完整的软文。

## 技术栈

- 后端: Flask
- 前端: HTML, CSS, JavaScript
- API: OpenAI

## 功能

- 生成标题: 根据用户输入的新闻素材 URL 和广告内容，生成多个软文标题及其对应的视角。
- 生成文章: 用户选择一个标题后，根据选定的标题和视角，生成一篇完整的软文。

## 安装与运行

1. 安装依赖
```bash
pip install -r requirements.txt
```

2.配置环境变量
```
OPENAIAPIKEY=youropenaiapikey
```

3.运行项目
```
python app.py
```

4.访问应用
在浏览器中打开 http://127.0.0.1:5000/ 来使用应用。

## 使用指南

输入新闻素材 URL: 在 “新闻素材 URL” 输入框中输入一个 URL。
输入广告内容: 在 “广告内容” 输入框中输入需要推广的产品或服务信息。
选择写作风格: 从下拉菜单中选择一个写作风格，例如 “新闻”、“公文”、“小红书” 或 “抖音口播”。
生成标题: 点击 “生成标题” 按钮，系统会根据输入的新闻素材和广告内容生成多个软文标题及其对应的视角。
选择标题: 点击生成的标题卡片，选择一个标题。
生成文章: 点击 “开始生成完整文章” 按钮，系统会根据选定的标题和视角生成一篇完整的软文。
复制文章: 点击 “复制内容” 按钮，可以将生成的文章内容复制到剪贴板。

## 演示
```html
<iframe width="560" height="315" src="https://www.youtube.com/embed/H062GSYAqts?si=_I4fwq9WYm8-4haE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
```
<iframe width="560" height="315" src="https://www.youtube.com/embed/H062GSYAqts?si=_I4fwq9WYm8-4haE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>