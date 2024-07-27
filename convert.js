// convert.js
const fs = require('fs');
const path = require('path');
const markdownIt = require('markdown-it');
const markdownItKatex = require('markdown-it-katex');
const md = markdownIt().use(markdownItKatex);

/**
 * 将 Markdown 文件转换为 HTML 文件
 * @param {string} filePath - Markdown 文件路径
 */
function convertMarkdownToHtml(filePath) {
  const markdownContent = fs.readFileSync(filePath, 'utf8');
  const htmlContent = md.render(markdownContent);
  const outputHtml = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${path.basename(filePath, '.md')}</title>
  <link rel="stylesheet" href="style.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.css">
</head>
<body>
  <div class="container">
    ${htmlContent}
  </div>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/contrib/auto-render.min.js"></script>
  <script>
    document.addEventListener("DOMContentLoaded", function() {
      renderMathInElement(document.body, {
        delimiters: [
          {left: "$$", right: "$$", display: true},
          {left: "$", right: "$", display: false},
          {left: "\\(", right: "\\)", display: false},
          {left: "\\[", right: "\\]", display: true}
        ]
      });
    });
  </script>
</body>
</html>
  `;
  const outputFilePath = filePath.replace(/\.md$/, '.html');
  fs.writeFileSync(outputFilePath, outputHtml);
  console.log(`Converted: ${filePath} -> ${outputFilePath}`);
}

/**
 * 遍历目录并转换所有 Markdown 文件
 * @param {string} dir - 起始目录
 * @param {Array} links - 保存 HTML 文件链接的数组
 */
function traverseDirectory(dir, links) {
  const files = fs.readdirSync(dir);
  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    if (stat.isDirectory()) {
      traverseDirectory(filePath, links);
    } else if (filePath.endsWith('.md')) {
      convertMarkdownToHtml(filePath);
      links.push(filePath.replace(/\.md$/, '.html'));
    }
  });
}

/**
 * 生成 index.html 文件
 * @param {Array} links - HTML 文件链接数组
 */
function generateIndexHtml(links) {
  const indexContent = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Index Page</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <div class="container">
    <h1>Index Page</h1>
    <ul>
      ${links.map(link => `<li><a href="${link}">${link}</a></li>`).join('\n')}
    </ul>
  </div>
</body>
</html>
  `;
  fs.writeFileSync('index.html', indexContent);
  console.log('Generated: index.html');
}

// 起始目录
const startDir = '.';
const links = [];
traverseDirectory(startDir, links);
generateIndexHtml(links);