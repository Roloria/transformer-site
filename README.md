# An Annotated Transformer

A visual, interactive field guide for systematically learning the Transformer
architecture — from tokens to residual streams, from softmax to scaled
dot-product attention.

## Features

- **Plain-English intro** — what a Transformer is, explained with analogies
- **16 structured sections** — Premise → Blueprint → Tokens → Attention →
  Playground → Multi-Head → Positions → FFN → Residual → Encoder/Decoder →
  Worked Example → Code → Variants → Glossary
- **Live attention playground** — type a sentence, watch it tokenize, click a
  token to see its attention pattern, Q/K/V matrices, and output vector
- **Worked example** — traces one sentence through the full pipeline with
  actual numbers at every step
- **Annotated PyTorch code** — a working MultiHeadAttention in 60 lines
- **KaTeX math** — formulas render beautifully in-browser
- **中英双语（i18n）** — 一键切换 EN / 中文，全站（含动态内容、进度侧栏、术语表）完整翻译，语言偏好自动保存

## Run locally

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

Or just open `index.html` in a browser.

## Stack

Vanilla HTML / CSS / JS. No build step. Fonts and KaTeX loaded from CDN.

## License

MIT
