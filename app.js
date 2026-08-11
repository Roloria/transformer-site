/* ===========================================================
   An Annotated Transformer — interactions
   =========================================================== */

/* ============ Utility: tiny seeded RNG ============ */
function rng(seed) {
  let s = seed | 0;
  return () => {
    s = (s * 1664525 + 1013904223) | 0;
    return ((s >>> 0) / 0xFFFFFFFF);
  };
}

/* ============ 1) HERO ATTENTION STRIP ============ */
(function buildAttentionStrip() {
  const svg = document.getElementById('stripLinks');
  if (!svg) return;
  const W = 600, H = 80;
  const tokens = svg.parentElement.querySelectorAll('.strip-token');
  const n = tokens.length;
  const r = rng(7);

  // For each token (as query), pick 2 strong attention targets
  for (let q = 0; q < n; q++) {
    const targets = new Set([q]);
    while (targets.size < 3) targets.add(Math.floor(r() * n));
    const x1 = (q + 0.5) * (W / n);
    targets.forEach(k => {
      if (k === q) return;
      const x2 = (k + 0.5) * (W / n);
      const weight = 0.25 + r() * 0.55;
      const opacity = weight * 0.7;
      const sw = 0.4 + weight * 1.4;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      const mx = (x1 + x2) / 2;
      const my = H * (0.2 + (q % 3) * 0.15);
      path.setAttribute('d', `M${x1},${H} Q${mx},${my} ${x2},${H}`);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', '#00d4ff');
      path.setAttribute('stroke-width', sw);
      path.setAttribute('opacity', opacity);
      path.setAttribute('stroke-linecap', 'round');
      svg.appendChild(path);
    });
  }
})();

/* ============ 2) STATIC HEATMAP (Attention QKᵀ) ============ */
(function buildStaticHeatmap() {
  const wrap = document.getElementById('scoreHeatmap');
  if (!wrap) return;
  const labels = ['The','cat','sat','on','the','mat','.','<EOS>'];
  const n = 8;
  // Generate a believable attention pattern: local + a few strong distant links
  const matrix = Array.from({length: n}, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    // local attention
    for (let j = 0; j <= i; j++) matrix[i][j] = 0.3;
    // strong specific links
    if (i === 2) matrix[i][1] = 1.0;       // sat -> cat (verb to subject)
    if (i === 4) matrix[i][1] = 0.95;      // the -> cat
    if (i === 5) matrix[i][4] = 1.0;       // mat -> the
    if (i === 5) matrix[i][2] = 0.7;       // mat -> sat
  }
  // softmax per row
  for (let i = 0; i < n; i++) {
    const row = matrix[i].slice(0, i + 1);
    const max = Math.max(...row);
    const exps = row.map(v => Math.exp((v - max) * 3));
    const sum = exps.reduce((a, b) => a + b, 0);
    for (let j = 0; j < i + 1; j++) matrix[i][j] = exps[j] / sum;
    for (let j = i + 1; j < n; j++) matrix[i][j] = 0;
  }
  wrap.innerHTML = '';
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const c = document.createElement('div');
      c.className = 'cell';
      c.style.setProperty('--o', matrix[i][j].toFixed(3));
      c.title = `${labels[i]} → ${labels[j]}: ${(matrix[i][j]*100).toFixed(1)}%`;
      wrap.appendChild(c);
    }
  }
  // column labels below
  const labelsWrap = document.createElement('div');
  labelsWrap.style.cssText = 'display:grid;grid-template-columns:repeat(8,1fr);gap:2px;margin-top:8px;font-family:JetBrains Mono;font-size:11px;color:#5a5954;text-align:center;';
  labels.forEach(l => {
    const s = document.createElement('span');
    s.textContent = l;
    labelsWrap.appendChild(s);
  });
  wrap.parentElement.appendChild(labelsWrap);
  const rowLabels = document.createElement('div');
  rowLabels.style.cssText = 'font-family:JetBrains Mono;font-size:11px;color:#5a5954;text-align:right;padding-right:8px;';
  // (row labels omitted to keep simple; hover tooltips show context)
})();

/* ============ 3) PLAYGROUND (interactive attention) ============ */
const PG = (function() {
  const input = document.getElementById('pgInput');
  const tokensEl = document.getElementById('pgTokens');
  const qMat = document.getElementById('pgQ');
  const kMat = document.getElementById('pgK');
  const vMat = document.getElementById('pgV');
  const heat = document.getElementById('pgHeatmap');
  const out = document.getElementById('pgOutput');

  const D_MODEL = 8;
  const D_K = 4;
  const HEADS = 1;

  // Simple character-level tokenizer
  function tokenize(s) {
    return s.trim().split(/\s+/).filter(Boolean);
  }

  // Deterministic embeddings: hash-based
  function hashEmbed(tok, dim) {
    let h = 2166136261;
    for (let i = 0; i < tok.length; i++) {
      h ^= tok.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      h = Math.imul(h ^ (h >>> 13), 1274126177);
      const x = ((h >>> 0) / 0xFFFFFFFF) - 0.5;
      v[i] = x * 1.6;
    }
    // L2-normalize
    let n = 0;
    for (let i = 0; i < dim; i++) n += v[i] * v[i];
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < dim; i++) v[i] /= n;
    return v;
  }

  // Fixed projection matrices (deterministic from constants)
  function makeMatrix(rows, cols, seed) {
    const r = rng(seed);
    const m = [];
    // Xavier-ish init
    const scale = Math.sqrt(1 / cols);
    for (let i = 0; i < rows; i++) {
      const row = new Float32Array(cols);
      for (let j = 0; j < cols; j++) row[j] = (r() - 0.5) * 2 * scale;
      m.push(row);
    }
    return m;
  }

  const WQ = makeMatrix(D_K * HEADS, D_MODEL, 101);
  const WK = makeMatrix(D_K * HEADS, D_MODEL, 202);
  const WV = makeMatrix(D_K * HEADS, D_MODEL, 303);

  function project(v, W) {
    const out = new Float32Array(W.length);
    for (let i = 0; i < W.length; i++) {
      let s = 0;
      const row = W[i];
      for (let j = 0; j < v.length; j++) s += row[j] * v[j];
      out[i] = s;
    }
    return out;
  }

  let tokens = [];
  let Q = [], K = [], V = [];
  let weights = [];   // softmax weights per row
  let selected = 0;

  function render() {
    const sentence = input.value || 'hello world';
    tokens = tokenize(sentence);
    Q = tokens.map(t => project(hashEmbed(t, D_MODEL), WQ));
    K = tokens.map(t => project(hashEmbed(t, D_MODEL), WK));
    V = tokens.map(t => project(hashEmbed(t, D_MODEL), WV));

    // scores = Q . K^T / sqrt(d_k)
    const scale = 1 / Math.sqrt(D_K);
    const scores = [];
    for (let i = 0; i < tokens.length; i++) {
      const row = new Float32Array(tokens.length);
      for (let j = 0; j < tokens.length; j++) {
        let s = 0;
        for (let k = 0; k < D_K; k++) s += Q[i][k] * K[j][k];
        row[j] = s * scale;
      }
      scores.push(row);
    }
    // softmax per row (causal mask — decoder-style)
    weights = [];
    for (let i = 0; i < tokens.length; i++) {
      const row = scores[i];
      const masked = row.map((v, j) => j <= i ? v : -Infinity);
      const max = Math.max(...masked.filter(v => isFinite(v)));
      const exps = masked.map(v => isFinite(v) ? Math.exp(v - max) : 0);
      const sum = exps.reduce((a, b) => a + b, 0);
      weights.push(exps.map(e => e / (sum || 1)));
    }

    // --- tokens row ---
    tokensEl.innerHTML = '';
    tokens.forEach((t, i) => {
      const span = document.createElement('span');
      span.className = 'pg-tok' + (i === selected ? ' is-selected' : '');
      span.textContent = t;
      span.onclick = () => { selected = i; render(); };
      tokensEl.appendChild(span);
    });

    // --- matrices (only show selected row highlighted) ---
    qMat.innerHTML = '<span class="m-tag q-tag">Q  (n × d_k)</span>';
    kMat.innerHTML = '<span class="m-tag k-tag">K  (n × d_k)</span>';
    vMat.innerHTML = '<span class="m-tag v-tag">V  (n × d_k)</span>';
    function fillMatrix(el, mat, cls) {
      for (let i = 0; i < mat.length; i++) {
        for (let j = 0; j < mat[i].length; j++) {
          const c = document.createElement('div');
          c.className = 'm-cell ' + cls;
          c.textContent = mat[i][j].toFixed(2);
          if (i === selected) c.style.background = 'rgba(0,212,255,0.18)';
          el.appendChild(c);
        }
      }
    }
    fillMatrix(qMat, Q, 'm-q');
    fillMatrix(kMat, K, 'm-k');
    fillMatrix(vMat, V, 'm-v');

    // --- heatmap ---
    heat.style.gridTemplateColumns = `repeat(${tokens.length}, 1fr)`;
    heat.innerHTML = '';
    for (let i = 0; i < tokens.length; i++) {
      for (let j = 0; j < tokens.length; j++) {
        const c = document.createElement('div');
        c.className = 'pg-cell';
        c.style.setProperty('--o', (weights[i][j] * 0.95 + 0.04).toFixed(3));
        if (j > i) c.style.opacity = '0.04';
        if (i === selected) c.classList.add('is-row-highlight');
        if (j === selected) c.classList.add('is-col-highlight');
        c.title = `${tokens[i]} → ${tokens[j]}: ${(weights[i][j]*100).toFixed(1)}%`;
        c.onclick = () => { selected = i; render(); };
        heat.appendChild(c);
      }
    }

    // --- output for selected ---
    out.innerHTML = '';
    const w = weights[selected] || [];
    const vec = new Array(D_K).fill(0);
    for (let j = 0; j < tokens.length; j++) {
      for (let k = 0; k < D_K; k++) vec[k] += w[j] * V[j][k];
    }
    vec.forEach((v, k) => {
      const s = document.createElement('span');
      s.textContent = `${k}: ${v.toFixed(2)}`;
      out.appendChild(s);
    });
    // small caption
    const cap = document.createElement('span');
    cap.style.cssText = 'width:100%;margin-top:6px;font-size:11px;color:#5a5954;';
    const capText = window.T ? window.T('pg.outputCaption').replace('{token}', tokens[selected]) : `Weighted sum of V vectors using attention weights from "${tokens[selected]}".`;
    cap.textContent = capText;
    out.appendChild(cap);
  }

  // wire up
  if (input) {
    input.addEventListener('input', () => { selected = 0; render(); });
    document.querySelectorAll('.pg-chip').forEach(btn => {
      btn.onclick = () => { input.value = btn.dataset.t; selected = 0; render(); };
    });
    render();
  }

  return { render };
})();

/* ============ 4) MULTI-HEAD ATTENTION CARDS ============ */
var HEADS = (function() {
  const grid = document.getElementById('headsGrid');
  if (!grid) return { rebuild: function(){} };
  const heads = [
    { nameKey: 'heads.h1name', descKey: 'heads.h1desc', color: '#00d4ff', pattern: [[1,.2,0,0,0],[.1,1,.3,.1,0],[0,.1,1,.2,0],[0,0,.2,1,.1],[0,0,0,.1,1]] },
    { nameKey: 'heads.h2name', descKey: 'heads.h2desc', color: '#9d4edd', pattern: [[.4,0,1,0,0],[0,.5,0,1,0],[1,0,.6,0,1],[0,1,0,.5,0],[.3,0,1,0,.7]] },
    { nameKey: 'heads.h3name', descKey: 'heads.h3desc', color: '#ff5722', pattern: [[1,0,0,0,0],[.8,1,0,0,0],[0,.8,1,0,0],[0,0,.8,1,0],[0,0,0,.8,1]] },
    { nameKey: 'heads.h4name', descKey: 'heads.h4desc', color: '#ffbe0b', pattern: [[1,.9,.1,.9,.1],[.9,1,.1,.9,.1],[.1,.1,1,.1,.1],[.9,.9,.1,1,.1],[.1,.1,.1,.1,1]] },
  ];
  function rebuild() {
    grid.innerHTML = '';
    heads.forEach((h, idx) => {
      const card = document.createElement('div');
      card.className = 'head-card';
      card.style.setProperty('--hc', h.color);
      const name = window.T ? window.T(h.nameKey) : h.nameKey;
      const desc = window.T ? window.T(h.descKey) : h.descKey;
      card.innerHTML = `
        <div class="head-id">HEAD ${String(idx + 1).padStart(2, '0')} · d_k = 64</div>
        <div class="head-name">${name}</div>
        <div class="head-desc">${desc}</div>
        <div class="head-vis">
          ${h.pattern.flat().map(v => `<div class="hv-cell" style="--o:${v.toFixed(2)}"></div>`).join('')}
        </div>
      `;
      grid.appendChild(card);
    });
  }
  rebuild();
  return { rebuild: rebuild };
})();

window.addEventListener('langchange', function() {
  if (window.HEADS && window.HEADS.rebuild) window.HEADS.rebuild();
});

/* ============ 5) POSITIONAL ENCODING HEATMAP (SVG) ============ */
(function buildPosHeat() {
  const svg = document.getElementById('posHeat');
  if (!svg) return;
  const W = 600, H = 200;
  const cols = 100, rows = 32;
  const cellW = W / cols, cellH = H / rows;
  const NS = 'http://www.w3.org/2000/svg';
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r;                          // dimension index
      const pos = c;
      const freq = 1 / Math.pow(10000, (2 * Math.floor(i / 2)) / rows);
      const angle = pos * freq;
      const v = (i % 2 === 0) ? Math.sin(angle) : Math.cos(angle);
      const intensity = Math.abs(v);
      const opacity = 0.08 + intensity * 0.75;
      const rect = document.createElementNS(NS, 'rect');
      rect.setAttribute('x', c * cellW);
      rect.setAttribute('y', r * cellH);
      rect.setAttribute('width', cellW);
      rect.setAttribute('height', cellH);
      rect.setAttribute('fill', '#ffbe0b');
      rect.setAttribute('opacity', opacity.toFixed(3));
      svg.appendChild(rect);
    }
  }
})();

/* ============ 6) CODE BLOCK with syntax highlight ============ */
(function buildCode() {
  const code = document.getElementById('codeBlock');
  const notes = document.getElementById('codeNotes');
  if (!code || !notes) return;

  const src = `import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # one big projection, then split into heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        B, T, _ = x.shape

        # 1. project into Q, K, V   (each: B, T, d_model)
        Q = self.W_q(x); K = self.W_k(x); V = self.W_v(x)

        # 2. split into h heads     (each: B, n_heads, T, d_k)
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 3. scaled dot-product    scores: (B, h, T_q, T_k)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. (optional) causal mask — set future positions to -inf
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5. softmax along the key dimension
        weights = scores.softmax(dim=-1)

        # 6. weighted sum of values  out: (B, h, T_q, d_k)
        out = weights @ V

        # 7. concatenate heads, final projection
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)`;

  // Very small token-based highlighter
  const PY_KW = new Set(['import','from','class','def','return','if','is','not','None','True','False','in','super','__init__','for','while','and','or','else','elif','self','lambda','with','as','try','except','raise','pass','break','continue','yield','global','nonlocal','assert','async','await']);
  const PY_BI = new Set(['torch','nn','math']);

  function escapeHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

  const lines = src.split('\n');
  let html = '';
  lines.forEach((line, idx) => {
    const ln = String(idx + 1).padStart(2, ' ') + '  ';
    // Tokenize per character: comments first, then strings, then keywords
    let lineHtml = '';
    let i = 0;
    while (i < line.length) {
      const ch = line[i];
      // comment
      if (ch === '#') {
        lineHtml += `<span class="cm">${escapeHtml(line.slice(i))}</span>`;
        i = line.length;
        continue;
      }
      // string
      if (ch === '"' || ch === "'") {
        const quote = ch;
        let j = i + 1;
        while (j < line.length && line[j] !== quote) {
          if (line[j] === '\\') j++;
          j++;
        }
        const end = Math.min(j + 1, line.length);
        lineHtml += `<span class="st">${escapeHtml(line.slice(i, end))}</span>`;
        i = end;
        continue;
      }
      // identifier / number
      if (/[A-Za-z_]/.test(ch)) {
        let j = i;
        while (j < line.length && /[A-Za-z0-9_]/.test(line[j])) j++;
        const word = line.slice(i, j);
        let cls = '';
        if (PY_KW.has(word)) cls = 'kw';
        else if (PY_BI.has(word)) cls = 'bi';
        else if (line[j] === '(') cls = 'fn';
        else if (word === 'self') cls = 'self';
        lineHtml += cls ? `<span class="${cls}">${escapeHtml(word)}</span>` : escapeHtml(word);
        i = j;
        continue;
      }
      if (/[0-9]/.test(ch)) {
        let j = i;
        while (j < line.length && /[0-9.]/.test(line[j])) j++;
        lineHtml += `<span class="nm">${escapeHtml(line.slice(i, j))}</span>`;
        i = j;
        continue;
      }
      if (/[+\-*/%=<>!&|]/.test(ch)) {
        lineHtml += `<span class="op">${escapeHtml(ch)}</span>`;
        i++;
        continue;
      }
      lineHtml += escapeHtml(ch);
      i++;
    }
    html += `<span class="ln">${ln}</span>${lineHtml}\n`;
  });
  code.innerHTML = html;

  const noteData = [
    { num: '01', bodyKey: 'notes.01' },
    { num: '02', bodyKey: 'notes.02' },
    { num: '03', bodyKey: 'notes.03' },
    { num: '04', bodyKey: 'notes.04' },
    { num: '05', bodyKey: 'notes.05' },
    { num: '06', bodyKey: 'notes.06' },
  ];
  function rebuildNotes() {
    notes.innerHTML = noteData.map(n => {
      const body = window.T ? window.T(n.bodyKey) : n.bodyKey;
      return `
        <div class="note">
          <div class="note-num">${n.num}</div>
          <div class="note-text">${body}</div>
        </div>
      `;
    }).join('');
  }
  rebuildNotes();
  window.addEventListener('langchange', rebuildNotes);
})();

/* ============ 7) REVEAL ON SCROLL ============ */
(function setupReveal() {
  const targets = [
    '.hero-title', '.hero-lede', '.hero-actions', '.attention-strip', '.hero-foot',
    '.section-head', '.compare', '.blueprint-wrap', '.blueprint-stats',
    '.token-flow', '.caption',
    '.formula-block', '.qkv', '.heatmap', '.softmax-eq',
    '.playground', '.heads-grid',
    '.pos-viz', '.ffn-vis', '.callout',
    '.residual-vis', '.split-grid', '.code-wrap', '.variants-grid',
    '.footer-inner'
  ];
  const els = document.querySelectorAll(targets.join(','));
  els.forEach(el => {
    if (el.matches('.section-head, .compare, .heads-grid, .split-grid, .variants-grid, .footer-cols, .qkv, .blueprint-stats')) {
      el.classList.add('reveal-stagger');
    } else {
      el.classList.add('reveal');
    }
  });

  const io = new IntersectionObserver((entries) => {
    entries.forEach(en => {
      if (en.isIntersecting) {
        en.target.classList.add('is-in');
        io.unobserve(en.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -10% 0px' });

  els.forEach(el => io.observe(el));
})();

/* ============ 8) KEYBOARD NAV ============ */
(function keys() {
  const ids = ['top','what','premise','blueprint','tokens','attention','playground','multihead','positions','ffn','residual','split','worked','code','variants','glossary','end'];
  let i = 0;

  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, textarea')) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'g' || e.key === 'G') {
      i = 0;
      document.getElementById('top').scrollIntoView({behavior: 'smooth'});
    } else if (e.key === 'j' || e.key === 'J') {
      i = Math.min(i + 1, ids.length - 1);
      document.getElementById(ids[i]).scrollIntoView({behavior: 'smooth'});
    } else if (e.key === 'k' || e.key === 'K') {
      i = Math.max(i - 1, 0);
      document.getElementById(ids[i]).scrollIntoView({behavior: 'smooth'});
    }
  });

  // keep index in sync
  window.addEventListener('scroll', () => {
    const y = window.scrollY + window.innerHeight / 3;
    for (let k = 0; k < ids.length; k++) {
      const el = document.getElementById(ids[k]);
      if (el && el.offsetTop <= y) i = k;
    }
  }, { passive: true });
})();

/* ============ 9) KATEX RENDER ============ */
window.addEventListener('load', () => {
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false }
      ],
      throwOnError: false
    });
  }
});

/* ============ PROGRESS RAIL ============ */
var PROGRESS = (function() {
  const sectionIds = ['top','what','premise','blueprint','tokens','attention','playground','multihead','positions','ffn','residual','split','worked','code','variants','glossary'];

  const list = document.getElementById('prList');
  const rail = document.getElementById('progressRail');
  const fill = document.getElementById('prBarFill');
  const count = document.getElementById('prCount');

  if (!list || !rail) return { rebuild: function(){} };

  function label(i) {
    return window.T ? window.T('rail.s' + i) : sectionIds[i];
  }

  function updateProgress() {
    const y = window.scrollY + window.innerHeight / 3;
    let currentIdx = 0;
    for (let i = 0; i < sectionIds.length; i++) {
      const el = document.getElementById(sectionIds[i]);
      if (el && el.offsetTop <= y) currentIdx = i;
    }

    // Update list items
    const items = list.querySelectorAll('li');
    items.forEach((li, i) => {
      li.classList.remove('is-current', 'is-done');
      if (i < currentIdx) li.classList.add('is-done');
      else if (i === currentIdx) {
        li.classList.add('is-current');
        // Scroll current item into view inside the list
        const listRect = list.getBoundingClientRect();
        const itemRect = li.getBoundingClientRect();
        if (itemRect.top < listRect.top || itemRect.bottom > listRect.bottom) {
          li.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
      }
    });

    // Update progress bar
    const pct = ((currentIdx + 1) / sectionIds.length) * 100;
    fill.style.width = pct + '%';

    // Update count
    count.textContent = (currentIdx + 1) + ' / ' + sectionIds.length;

    // Show rail after scrolling past hero
    if (window.scrollY > 600) {
      rail.classList.add('is-visible');
    } else {
      rail.classList.remove('is-visible');
    }
  }

  function rebuild() {
    list.innerHTML = sectionIds.map((id, i) =>
      `<li data-target="${id}" title="${label(i)}">${label(i)}</li>`
    ).join('');
    list.querySelectorAll('li').forEach(li => {
      li.onclick = () => {
        const el = document.getElementById(li.dataset.target);
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      };
    });
    updateProgress();
  }

  window.addEventListener('scroll', updateProgress, { passive: true });
  rebuild();
  return { rebuild: rebuild };
})();

// Rebuild progress rail on language change
window.addEventListener('langchange', function() {
  if (window.PROGRESS && window.PROGRESS.rebuild) window.PROGRESS.rebuild();
});

(function setupMobile() {
  const mob = document.createElement('div');
  mob.className = 'mobile-progress';
  mob.innerHTML = '<div class="mobile-progress-fill" id="mobFill"></div>';
  document.body.appendChild(mob);

  const mobFill = document.getElementById('mobFill');
  function updateMobile() {
    const total = document.body.scrollHeight - window.innerHeight;
    const pct = total > 0 ? (window.scrollY / total) * 100 : 0;
    if (mobFill) mobFill.style.width = pct + '%';
  }
  window.addEventListener('scroll', updateMobile, { passive: true });
})();
