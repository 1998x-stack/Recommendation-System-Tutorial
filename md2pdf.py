import os
import re
import markdown2
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem

def register_fonts():
    """
    注册中文字体。
    """
    pdfmetrics.registerFont(TTFont('ArialUnicode', 'ArialUnicode.ttf'))

def remove_html_tags(text: str) -> str:
    """
    去除文本中的HTML标签。

    :param text: 包含HTML标签的文本。
    :return: 去除HTML标签后的纯文本。
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def ensure_style(styles, style_name, **kwargs):
    """
    确保样式表中没有重复样式定义。

    :param styles: 样式表对象。
    :param style_name: 样式名称。
    :param kwargs: 样式属性。
    """
    if style_name not in styles:
        styles.add(ParagraphStyle(name=style_name, **kwargs))

def markdown_to_pdf(markdown_file_path: str, output_pdf_path: str) -> None:
    """
    将Markdown文件转换为PDF文件。

    :param markdown_file_path: 输入的Markdown文件路径。
    :param output_pdf_path: 输出的PDF文件路径。
    """
    # 读取Markdown文件内容
    with open(markdown_file_path, 'r', encoding='utf-8') as markdown_file:
        markdown_content = markdown_file.read()

    # 将Markdown转换为HTML
    html_content = markdown2.markdown(markdown_content)

    # 创建PDF文档
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()

    # 确保不会重复定义样式
    ensure_style(styles, 'Chinese', fontName='ArialUnicode', fontSize=12)
    ensure_style(styles, 'Heading1', parent=styles['Heading1'], fontSize=18, spaceAfter=6, fontName='ArialUnicode')
    ensure_style(styles, 'Heading2', parent=styles['Heading2'], fontSize=16, spaceAfter=6, fontName='ArialUnicode')
    ensure_style(styles, 'Heading3', parent=styles['Heading3'], fontSize=14, spaceAfter=6, fontName='ArialUnicode')
    ensure_style(styles, 'Bold', parent=styles['Normal'], fontName='ArialUnicode', fontSize=12, textColor='black', bold=True)
    ensure_style(styles, 'Italic', parent=styles['Normal'], fontName='ArialUnicode', fontSize=12, textColor='black', italic=True)

    # 创建内容列表
    content = []

    # 处理Markdown内容
    def process_markdown_paragraph(paragraph: str):
        if paragraph.startswith('# '):
            return Paragraph(paragraph[2:], styles['Heading1'])
        elif paragraph.startswith('## '):
            return Paragraph(paragraph[3:], styles['Heading2'])
        elif paragraph.startswith('### '):
            return Paragraph(paragraph[4:], styles['Heading3'])
        elif paragraph.startswith('* ') or paragraph.startswith('- '):
            items = [remove_html_tags(item.strip()) for item in paragraph.split('\n') if item.strip()]
            return ListFlowable([ListItem(Paragraph(item, styles['Chinese'])) for item in items], bulletType='bullet')
        elif '**' in paragraph:
            parts = re.split(r'(\*\*.*?\*\*)', paragraph)
            return Paragraph(''.join(f"<b>{part[2:-2]}</b>" if part.startswith('**') and part.endswith('**') else part for part in parts), styles['Chinese'])
        elif '*' in paragraph:
            parts = re.split(r'(\*.*?\*)', paragraph)
            return Paragraph(''.join(f"<i>{part[1:-1]}</i>" if part.startswith('*') and part.endswith('*') else part for part in parts), styles['Chinese'])
        else:
            return Paragraph(paragraph, styles['Chinese'])

    # 将HTML内容按段落拆分并添加到PDF内容中
    paragraphs = html_content.split('\n')
    for paragraph in paragraphs:
        paragraph = remove_html_tags(paragraph)
        if paragraph.strip() != '':
            content.append(process_markdown_paragraph(paragraph))
            content.append(Spacer(1, 12))  # 添加段落间距

    # 生成PDF文档
    doc.build(content)

def convert_markdown_in_directory(root_dir: str) -> None:
    """
    将目录及其子目录中的所有Markdown文件转换为PDF文件。

    :param root_dir: 要搜索Markdown文件的根目录。
    """
    register_fonts()
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.md'):
                markdown_path = os.path.join(dirpath, filename)
                pdf_path = os.path.splitext(markdown_path)[0] + '.pdf'
                print(f'Converting {markdown_path} to {pdf_path}')
                markdown_to_pdf(markdown_path, pdf_path)

# 示例调用
root_directory = './'  # 替换为你要搜索的根目录
convert_markdown_in_directory(root_directory)
