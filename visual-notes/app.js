/* 케라스 딥러닝 공책 — 렌더러
 * 데이터: data/chNN.js 가 BOOK.add({...}) 로 장을 등록한다.
 * 카드 형식(내 필기 문법):
 *   title  : 주황 형광펜 제목
 *   fig    : { h, f: k => [ ...k.프리미티브() ] }  → 연두 형광펜 박스 안 그림
 *   pts    : ["· 줄", "  · 들여쓴 줄"]  ==민트 밑줄==  !!빨간 펜!!  `코드`  **굵게**
 *   end    : 빨간 펜 결론 한 줄
 *   mine   : { img, date }  → 내가 공책에 쓴 필기 사진
 */
(function () {
  const BOOK = (window.BOOK = { chapters: [], labs: {}, add(ch) { this.chapters.push(ch); }, addLab(l) { this.labs[l.ch] = l; } });
  const W = 760;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  /* ---------- SVG 프리미티브 ---------- */
  const textW = (s, size) => {
    let w = 0;
    for (const ch of String(s)) w += /[ㄱ-힝]/.test(ch) ? size * 0.98 : size * 0.58;
    return w;
  };
  function T(x, y, s, o = {}) {
    const size = o.size || 15, lines = String(s).split("\n"), lh = size * 1.28;
    const anchor = o.anchor || "middle";
    const cls = ["t", o.color ? "t-" + o.color : "", o.hand ? "t-hand" : "", o.mono ? "t-mono" : "", o.bold ? "t-bold" : ""].join(" ");
    const y0 = y - ((lines.length - 1) * lh) / 2 + size * 0.36;
    let out = "";
    if (o.u) {
      lines.forEach((l, i) => {
        const w = textW(l, size), ly = y0 + i * lh + 4;
        const x0 = anchor === "middle" ? x - w / 2 : anchor === "end" ? x - w : x;
        out += `<line class="uline" x1="${x0 - 2}" y1="${ly}" x2="${x0 + w + 2}" y2="${ly}"/>`;
      });
    }
    out += `<text class="${cls}" x="${x}" y="${y0}" font-size="${size}" text-anchor="${anchor}">` +
      lines.map((l, i) => `<tspan x="${x}" dy="${i ? lh : 0}">${esc(l)}</tspan>`).join("") + `</text>`;
    return out;
  }
  const sc = o => `sh f-${o.fill || "paper"}${o.dash ? " dash" : ""}${o.color ? " s-" + o.color : ""}${o.thick ? " thick" : ""}`;

  const K = {
    text: (x, y, s, o = {}) => T(x, y, s, Object.assign({ anchor: "start" }, o)),
    ctext: (x, y, s, o = {}) => T(x, y, s, o),
    red: (x, y, s, o = {}) => T(x, y, s, Object.assign({ anchor: "start", color: "red", hand: true, size: 19 }, o)),
    box(x, y, w, h, label, o = {}) {
      let s = `<rect class="${sc(o)}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${o.r ?? 4}"/>`;
      if (label != null && label !== "") {
        const cy = o.sub ? y + h / 2 - 8 : y + h / 2;
        s += T(x + w / 2, cy, label, { size: o.size || 15, bold: o.bold, color: o.tc, mono: o.mono, u: o.u });
      }
      if (o.sub) s += T(x + w / 2, y + h / 2 + 13, o.sub, { size: o.ss || 12, color: o.subc || "muted", mono: o.submono });
      return s;
    },
    oval(cx, cy, rx, ry, label, o = {}) {
      let s = `<ellipse class="${sc(o)}" cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}"/>`;
      if (label != null) s += T(cx, o.sub ? cy - 7 : cy, label, { size: o.size || 15, bold: o.bold, color: o.tc });
      if (o.sub) s += T(cx, cy + 12, o.sub, { size: 12, color: o.subc || "muted" });
      return s;
    },
    circle(cx, cy, r, label, o = {}) {
      let s = `<circle class="${sc(o)}" cx="${cx}" cy="${cy}" r="${r}"/>`;
      if (label != null) s += T(cx, cy, label, { size: o.size || 14, bold: o.bold, color: o.tc });
      return s;
    },
    dot: (x, y, o = {}) => `<circle class="dot d-${o.color || "ink"}" cx="${x}" cy="${y}" r="${o.r || 4}"/>`,
    line: (x1, y1, x2, y2, o = {}) =>
      `<line class="ln s-${o.color || "ink"}${o.dash ? " dash" : ""}${o.thin ? " thin" : ""}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`,
    arrow(x1, y1, x2, y2, o = {}) {
      const c = o.color || "ink";
      let s = `<line class="ln s-${c}${o.dash ? " dash" : ""}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#ah-${c})"${o.both ? ` marker-start="url(#ah-${c})"` : ""}/>`;
      if (o.label) s += T((x1 + x2) / 2 + (o.lx || 0), (y1 + y2) / 2 + (o.ly ?? -11), o.label, { size: o.ls || 12, color: o.lc || "muted", hand: o.hand });
      return s;
    },
    path(d, o = {}) {
      const c = o.color || "ink";
      let s = `<path class="ln s-${c}${o.dash ? " dash" : ""}${o.thick ? " thick" : ""}" d="${d}"${o.arrow === false ? "" : ` marker-end="url(#ah-${c})"`}${o.both ? ` marker-start="url(#ah-${c})"` : ""}${o.fill ? ` style="fill:var(--${o.fill})"` : ""}/>`;
      if (o.label) s += T(o.at[0], o.at[1], o.label, { size: o.ls || 12, color: o.lc || "muted", anchor: o.anchor, hand: o.hand });
      return s;
    },
    /* 형광펜·루프 묶음 박스 */
    frame(x, y, w, h, o = {}) {
      let s = `<rect class="frame fr-${o.color || "mint"}${o.dash ? " dash" : ""}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${o.r ?? 8}"/>`;
      if (o.label) s += T(x + 10, y + 14, o.label, { anchor: "start", size: 13, color: o.lc || "teal", bold: true });
      return s;
    },
    hl: (x, y, w, h, c = "yellow") => `<rect class="mark m-${c}" x="${x}" y="${y}" width="${w}" height="${h}" rx="3"/>`,
    /* 벡터 칸 */
    cells(x, y, vals, o = {}) {
      const cw = o.cw || 34, ch = o.ch || cw;
      return vals.map((v, i) => {
        const f = o.hi && o.hi.includes(i) ? o.hic || "yellow" : o.fill || "paper";
        return `<rect class="sh f-${f}" x="${x + i * cw}" y="${y}" width="${cw}" height="${ch}"/>` +
          (v === "" || v == null ? "" : T(x + i * cw + cw / 2, y + ch / 2, v, { size: o.size || 14, mono: o.mono, color: o.tc }));
      }).join("");
    },
    /* 행렬 */
    grid(x, y, r, c, vals, o = {}) {
      const cw = o.cw || 30, ch = o.ch || cw;
      let s = "";
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
        const k = i * c + j, v = vals ? vals[k] : null;
        let f = o.fill || "paper", st = "";
        if (o.hi && o.hi.includes(k)) f = o.hic || "yellow";
        if (o.heat) { f = o.heatc || "mint"; st = ` style="fill-opacity:${0.08 + 0.92 * o.heat[k]}"`; }
        s += `<rect class="sh f-${f}${o.thin ? " thin" : ""}" x="${x + j * cw}" y="${y + i * ch}" width="${cw}" height="${ch}"${st}/>`;
        if (v != null && v !== "") s += T(x + j * cw + cw / 2, y + i * ch + ch / 2, v, { size: o.size || 13, mono: o.mono, color: o.tc });
      }
      if (o.box) s += `<rect class="sh f-none thick s-${o.box}" x="${x + o.box_[1] * cw}" y="${y + o.box_[0] * ch}" width="${o.box_[3] * cw}" height="${o.box_[2] * ch}"/>`;
      return s;
    },
    /* 겹친 행렬 = 랭크-3 텐서 */
    cube(x, y, r, c, d, o = {}) {
      const dx = o.dx ?? 13, dy = o.dy ?? -13;
      let s = "";
      for (let k = d - 1; k >= 0; k--) s += K.grid(x + k * dx, y + k * dy, r, c, k === 0 ? o.vals : null, Object.assign({}, o, { fill: k === 0 ? o.fill : o.backfill || o.fill }));
      return s;
    },
    /* 작은 신경망 */
    nn(x, y, w, h, layers, o = {}) {
      const xs = layers.map((_, i) => x + (layers.length === 1 ? w / 2 : (i * w) / (layers.length - 1)));
      const pos = layers.map((n, i) => Array.from({ length: n }, (_, j) => [xs[i], y + (n === 1 ? h / 2 : (j * h) / (n - 1))]));
      const drop = new Set(o.drop || []);
      let s = "";
      for (let i = 0; i < pos.length - 1; i++)
        for (let a = 0; a < pos[i].length; a++) for (let b = 0; b < pos[i + 1].length; b++) {
          const dead = drop.has(`${i}-${a}`) || drop.has(`${i + 1}-${b}`);
          s += `<line class="ln thin s-${dead ? "faint" : "muted"}" x1="${pos[i][a][0]}" y1="${pos[i][a][1]}" x2="${pos[i + 1][b][0]}" y2="${pos[i + 1][b][1]}"/>`;
        }
      const r = o.r || 8;
      pos.forEach((col, i) => col.forEach((p, j) => {
        const dead = drop.has(`${i}-${j}`);
        s += `<circle class="sh f-${dead ? "paper" : (o.fills && o.fills[i]) || "paper"}${dead ? " dash" : ""}" cx="${p[0]}" cy="${p[1]}" r="${r}"/>`;
        if (dead) s += `<path class="ln s-red" d="M${p[0] - r} ${p[1] - r}L${p[0] + r} ${p[1] + r}M${p[0] + r} ${p[1] - r}L${p[0] - r} ${p[1] + r}"/>`;
      }));
      if (o.labels) o.labels.forEach((l, i) => l && (s += T(xs[i], y + h + 22, l, { size: 12, color: "muted" })));
      return s;
    },
    /* 그래프: 곡선 f(t) t∈[0,1] → [0,1] */
    plot(x, y, w, h, o = {}) {
      const X = t => x + t * w, Y = v => y + h - v * h;
      let s = `<path class="ln s-ink" d="M${x} ${y - 6}V${y + h}H${x + w + 6}" marker-end="url(#ah-ink)"/>`;
      s += `<line class="ln s-ink" x1="${x}" y1="${y - 4}" x2="${x}" y2="${y - 6}" marker-end="url(#ah-ink)"/>`;
      if (o.xl) s += T(x + w + 4, y + h + 18, o.xl, { size: 12, color: "muted", anchor: "end" });
      if (o.yl) s += T(x + 6, y - 2, o.yl, { size: 12, color: "muted", anchor: "start" });
      (o.bands || []).forEach(b => {
        s += `<rect class="mark m-${b.c || "yellow"}" x="${X(b.from)}" y="${y}" width="${X(b.to) - X(b.from)}" height="${h}"/>`;
        if (b.label) s += T((X(b.from) + X(b.to)) / 2, y + 12, b.label, { size: 12, color: b.lc || "muted" });
      });
      (o.curves || []).forEach(c => {
        const n = 80, d = Array.from({ length: n + 1 }, (_, i) => { const t = (c.from ?? 0) + ((c.to ?? 1) - (c.from ?? 0)) * (i / n); return `${i ? "L" : "M"}${X(t).toFixed(1)} ${Y(c.f(t)).toFixed(1)}`; }).join("");
        s += `<path class="ln s-${c.color || "ink"}${c.dash ? " dash" : ""}${c.thick ? " thick" : ""}" d="${d}"/>`;
        if (c.label) { const t = c.lt ?? (c.to ?? 1); s += T(X(t) + (c.lx || 6), Y(c.f(t)) + (c.ly || 0), c.label, { size: 12, anchor: c.anchor || "start", color: c.lc || c.color || "ink", hand: c.hand }); }
      });
      (o.vlines || []).forEach(v => {
        s += `<line class="ln dash s-${v.color || "red"}" x1="${X(v.t)}" y1="${y}" x2="${X(v.t)}" y2="${y + h}"/>`;
        if (v.label) s += T(X(v.t), y + h + 18, v.label, { size: 12, color: v.color || "red" });
      });
      (o.pts || []).forEach(p => {
        s += `<circle class="sh f-${p[3] || "ink"}" cx="${X(p[0])}" cy="${Y(p[1])}" r="${p[4] || 5}"/>`;
        if (p[2]) s += T(X(p[0]) + 8, Y(p[1]) - 10, p[2], { size: 12, anchor: "start", color: p[5] || "ink" });
      });
      return s;
    },
    bars(x, y, w, h, vals, o = {}) {
      const n = vals.length, g = o.gap ?? 4, bw = (w - g * (n - 1)) / n;
      let s = `<line class="ln s-ink" x1="${x - 2}" y1="${y + h}" x2="${x + w + 2}" y2="${y + h}"/>`;
      vals.forEach((v, i) => {
        const bh = v * h, bx = x + i * (bw + g);
        s += `<rect class="sh f-${o.hi && o.hi.includes(i) ? o.hic || "red" : o.fill || "mint"}" x="${bx}" y="${y + h - bh}" width="${bw}" height="${bh}"/>`;
        if (o.labels) s += T(bx + bw / 2, y + h + 13, o.labels[i], { size: o.ls || 11, color: "muted" });
      });
      return s;
    },
    /* 왼→오 흐름: 항목 = "글자" 또는 {t, sub, fill, shape:'oval', a:'다음 화살표 라벨'} */
    flow(y, items, o = {}) {
      const x0 = o.x0 ?? 20, x1 = o.x1 ?? 740, g = o.gap ?? 40, h = o.h ?? 50, n = items.length;
      const w = (x1 - x0 - (n - 1) * g) / n;
      let s = "";
      items.forEach((it, i) => {
        const t = typeof it === "string" ? { t: it } : it, x = x0 + i * (w + g);
        const op = Object.assign({ size: o.size, ss: o.ss }, t);
        s += (t.shape || o.shape) === "oval" ? K.oval(x + w / 2, y + h / 2, w / 2, h / 2, t.t, op) : K.box(x, y, w, h, t.t, op);
        if (i < n - 1) s += K.arrow(x + w + 3, y + h / 2, x + w + g - 3, y + h / 2, { label: t.a, color: o.ac });
      });
      return s;
    },
    raw: s => s,
  };

  /* ---------- 마크업 ---------- */
  function fmt(s) {
    return esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/==([^=]+)==/g, '<span class="u">$1</span>')
      .replace(/!!([^!]+)!!/g, '<span class="pen">$1</span>')
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  }
  function figSVG(fig) {
    const body = [].concat(fig.f(K)).join("");
    return `<svg class="fig" viewBox="0 0 ${fig.w || W} ${fig.h}" role="img" aria-label="${esc(fig.alt || "")}">${body}</svg>`;
  }
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch (e) {} },
  };

  function cardHTML(card, opt = {}) {
    const pts = (card.pts || []).map(line => {
      const lvl = Math.floor(line.match(/^ */)[0].length / 2);
      const t = line.trim().replace(/^·\s*/, "");
      return `<div class="pt l${lvl}">${fmt(t)}</div>`;
    }).join("");
    const done = store.get("copied:" + card.id) === "1";
    const mine = !card.mine ? "" : opt.compact
      ? `<a class="mine-badge" href="${esc(card.mine.img)}" target="_blank" rel="noopener">내 필기 ✓</a>`
      : `<a class="mine-badge" href="#" data-jump="m-${esc(card.id)}">내 필기 ✓</a>`;
    const photo = card.mine && !opt.compact
      ? `<figure class="photo mine-photo" id="m-${esc(card.id)}"><img src="${esc(card.mine.img)}" alt="${esc(card.title)} 손필기 사진" loading="lazy"><figcaption>내 공책 · ${esc(card.title)} · ${esc(card.mine.date || "")}</figcaption></figure>`
      : "";
    return `<article class="page${opt.compact ? " compact" : ""}" id="c-${esc(card.id)}">
      <header class="page-head"><span class="sec" title="책 위치">${esc(String(card.sec).split(".")[0])}장 · §${esc(card.sec)}</span><h3 class="title"><span>${esc(card.title)}</span></h3>${mine}</header>
      <div class="hl-box">${figSVG(card.fig)}</div>
      ${pts ? `<div class="pts">${pts}</div>` : ""}
      ${card.end ? `<p class="end">${fmt(card.end)}</p>` : ""}
      ${photo}
      ${opt.compact ? "" : `<footer class="page-foot"><label class="copy"><input type="checkbox" id="cp-${esc(card.id)}" data-id="${esc(card.id)}"${done ? " checked" : ""}> 공책에 옮겨 적음</label>${card.mine ? `<span class="foot-mine">내 필기 ${esc(card.mine.date || "")}</span>` : `<span class="foot-todo">사진 올리면 여기에 내 필기가 붙어요</span>`}</footer>`}
    </article>`;
  }

  /* ---------- 화면 ---------- */
  const $ = s => document.querySelector(s);
  const allCards = () => BOOK.chapters.flatMap(c => c.cards.map(k => Object.assign({ ch: c.no }, k)));

  function nav(active) {
    const items = [`<a href="#home" class="${active === "home" ? "on" : ""}"><span class="n">★</span><span class="nt">처음 / 내 필기</span></a>`]
      .concat(BOOK.chapters.map(c => {
        const mine = c.cards.filter(k => k.mine).length;
        return `<a href="#ch${String(c.no).padStart(2, "0")}" class="${active === c.no ? "on" : ""}"><span class="n">${c.no}</span><span class="nt">${esc(c.short || c.title)}</span><span class="cnt">${mine ? `<i class="m">${mine}</i>` : ""}${c.cards.length}</span></a>`;
      }));
    $("#nav").innerHTML = items.join("");
  }

  function chapterView(no) {
    const i = BOOK.chapters.findIndex(c => c.no === no), c = BOOK.chapters[i];
    if (!c) return homeView();
    nav(no);
    const prev = BOOK.chapters[i - 1], next = BOOK.chapters[i + 1];
    const mine = c.cards.filter(k => k.mine).length;
    $("#main").innerHTML = `<section class="ch-head">
        <div class="ch-no">${c.no}장</div>
        <h2>${esc(c.title)}</h2>
        <p class="ch-one">${fmt(c.one)}</p>
        <div class="ch-meta">그림 ${c.cards.length}장 · 내 필기 ${mine}장 · <span id="copied-count"></span></div>
        ${BOOK.labs[c.no] ? `<a class="lab-link" href="#ch${String(c.no).padStart(2, "0")}-lab"><b>실습 노트북</b><span>코드와 실행 결과를 같이 보기 →</span></a>` : ""}
        <div class="toc">${c.cards.map(k => `<a href="#ch${String(c.no).padStart(2, "0")}" data-jump="c-${esc(k.id)}">${esc(k.sec)} ${esc(k.title)}${k.mine ? " ✓" : ""}</a>`).join("")}</div>
      </section>
      ${c.terms ? `<section class="terms" aria-label="${esc(c.terms.title)}"><h3 class="terms-title"><span>${esc(c.terms.title)}</span></h3><dl>${c.terms.items.map(([t, d]) => `<div class="term${/^\s/.test(t) ? " sub" : ""}"><dt>${esc(t.trim().replace(/^·\s*/, ""))}</dt><dd>${fmt(d)}</dd></div>`).join("")}</dl></section>` : ""}
      <div class="pages">${c.cards.map(k => cardHTML(k)).join("")}</div>
      <nav class="pager">${prev ? `<a href="#ch${String(prev.no).padStart(2, "0")}">← ${prev.no}장 ${esc(prev.short || prev.title)}</a>` : "<span></span>"}${next ? `<a href="#ch${String(next.no).padStart(2, "0")}">${next.no}장 ${esc(next.short || next.title)} →</a>` : "<span></span>"}</nav>`;
    updateCopied(c);
  }
  /* ---------- 실습 (노트북 실행 기록) ---------- */
  function hlPy(src) {
    return src.split("\n").map(line => {
      let q = null, cut = -1;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (q) { if (ch === "\\") i++; else if (ch === q) q = null; }
        else if (ch === '"' || ch === "'") q = ch;
        else if (ch === "#") { cut = i; break; }
      }
      return cut < 0 ? esc(line) : esc(line.slice(0, cut)) + `<span class="cm">${esc(line.slice(cut))}</span>`;
    }).join("\n");
  }
  function labView(no) {
    const c = BOOK.chapters.find(x => x.no === no), lab = BOOK.labs[no];
    if (!c || !lab) return chapterView(no);
    nav(no);
    const nn = String(no).padStart(2, "0");
    let n = 0;
    const body = lab.cells.map(k => {
      if (k.t === "md") return `<div class="lab-md">${k.html}</div>`;
      n++;
      const out = k.out || k.imgs.length
        ? `<div class="lab-out"><span class="lab-tag">결과</span>${k.out ? `<pre>${esc(k.out)}</pre>` : ""}${k.imgs.map(u => `<img src="${esc(u)}" alt="셀 ${n} 실행 결과 그래프" loading="lazy">`).join("")}</div>`
        : "";
      return `<div class="lab-cell"><div class="lab-in"><span class="lab-tag">[${n}]</span><pre><code>${hlPy(k.src)}</code></pre></div>${out}</div>`;
    }).join("");
    $("#main").innerHTML = `<section class="ch-head">
        <a class="lab-back" href="#ch${nn}">← ${c.no}장 그림 노트로</a>
        <div class="ch-no">${c.no}장 · 실습</div>
        <h2>${esc(lab.title)}</h2>
        <div class="ch-meta">실행 기록: ${esc(lab.run)} · 원본 <code>${esc(lab.file)}</code></div>
      </section>
      <article class="lab">${body}</article>
      <nav class="pager"><a href="#ch${nn}">← ${c.no}장 그림 노트로</a><span></span></nav>`;
  }

  function updateCopied(c) {
    const el = $("#copied-count");
    if (!el) return;
    const n = c.cards.filter(k => store.get("copied:" + k.id) === "1").length;
    el.textContent = `옮겨 적음 ${n}/${c.cards.length}`;
  }

  function homeView() {
    nav("home");
    const cards = allCards(), mine = cards.filter(k => k.mine);
    const grammar = [
      ["title", "제목은 주황 형광펜", "개념 이름 하나. 문장 아님."],
      ["box", "핵심 그림은 연두 형광펜 박스", "그림이 먼저, 글은 그 아래."],
      ["flow", "흐름은 → 로", "입력·출력은 타원, 처리는 사각형."],
      ["num", "· 불릿 + ① ② 로 쪼개기", "\"용어 : 짧은 정의\" 형태."],
      ["u", "키워드는 민트 밑줄", "다시 볼 때 눈이 먼저 가는 곳."],
      ["pen", "빨간 펜 = 내 말", "주석, 그리고 \"*결국 ~이 본질\" 결론."],
    ];
    $("#main").innerHTML = `<section class="hero">
        <p class="eyebrow">케라스 창시자에게 배우는 딥러닝 개정 3판</p>
        <h1><span>케라스 딥러닝 공책</span></h1>
        <p class="lede">책의 긴 글을 <span class="u">내 필기 형식</span>으로 뜯어서 그림 ${cards.length}장으로 만들었어요. 그림을 보고 공책에 옮겨 그리면서 이해하고, 다 쓴 필기는 사진으로 올려요.</p>
        <div class="stats"><div><b>${BOOK.chapters.length}</b><span>장</span></div><div><b>${cards.length}</b><span>그림</span></div><div><b>${mine.length}</b><span>내 필기</span></div></div>
      </section>

      <section class="block">
        <h2 class="bh"><span>내 필기 → 정리본</span></h2>
        <p class="bp">직접 쓴 필기와, 같은 내용을 같은 문법으로 다시 그린 정리본이에요. <span class="pen">빨간 글씨 중 "+책"</span> 표시는 책에서 보충한 부분이에요.</p>
        ${mine.map(k => `<div class="pair">
            <figure class="photo"><a href="${esc(k.mine.img)}" target="_blank" rel="noopener"><img src="${esc(k.mine.img)}" alt="${esc(k.title)} 손필기 사진" loading="lazy"></a><figcaption>내 공책 · ${esc(k.title)} · ${esc(k.mine.date || "")}</figcaption></figure>
            <div class="redraw">${cardHTML(k, { compact: true })}<a class="goto" href="#ch${String(k.ch).padStart(2, "0")}">${k.ch}장에서 보기 →</a></div>
          </div>`).join("")}
      </section>

      <section class="block">
        <h2 class="bh"><span>내 필기 문법</span></h2>
        <p class="bp">필기 두 장에서 뽑은 규칙이에요. 모든 그림이 이 규칙대로 그려져 있어요.</p>
        <ol class="grammar">${grammar.map(([k, h, p]) => `<li><div class="gx gx-${k}">${{
          title: '<span class="t-mark">텐서</span>',
          box: '<span class="g-box">[1]→[1|2]→…</span>',
          flow: '<span class="g-oval">입력</span> → <span class="g-rect">층</span> → <span class="g-oval">예측</span>',
          num: '· 관점 ① … ② …',
          u: '<span class="u">손실 함수</span>',
          pen: '<span class="pen">*결국, 가중치 조정이 본질</span>',
        }[k]}</div><b>${h}</b><span>${p}</span></li>`).join("")}</ol>
      </section>

      <section class="block">
        <h2 class="bh"><span>공부 루프</span></h2>
        <ol class="loop">
          <li><b>그림 고르기</b><span>장을 열고 그림 한 장을 골라요.</span></li>
          <li><b>공책에 옮겨 그리기</b><span>그림 먼저, 불릿은 내 말로 줄여 써요. 다 쓰면 "공책에 옮겨 적음" 체크.</span></li>
          <li><b>사진 올리기</b><span><code>notes/</code> 폴더에 사진을 넣어요.</span></li>
          <li><b>Claude에게 "등록해줘"</b><span>해당 그림에 <span class="mine-inline">내 필기 ✓</span>가 붙고, 빠진 부분이나 틀린 부분이 있으면 알려줘요.</span></li>
        </ol>
      </section>

      <section class="block">
        <h2 class="bh"><span>목차</span></h2>
        <div class="chgrid">${BOOK.chapters.map(c => {
          const m = c.cards.filter(k => k.mine).length;
          return `<a href="#ch${String(c.no).padStart(2, "0")}" class="chcard"><span class="cn">${c.no}장</span><b>${esc(c.title)}</b><span class="co">${fmt(c.one)}</span><span class="cm">그림 ${c.cards.length}${m ? ` · <i>내 필기 ${m}</i>` : ""}</span></a>`;
        }).join("")}</div>
      </section>`;
  }

  function route() {
    const h = location.hash.replace("#", "");
    const m = h.match(/^ch(\d\d)(-lab)?$/);
    if (m) (m[2] ? labView : chapterView)(+m[1]); else homeView();
    window.scrollTo(0, 0);
  }

  document.addEventListener("change", e => {
    const t = e.target;
    if (t.matches && t.matches('.copy input[type="checkbox"]')) {
      store.set("copied:" + t.dataset.id, t.checked ? "1" : null);
      const m = location.hash.match(/^#ch(\d\d)$/);
      if (m) updateCopied(BOOK.chapters.find(c => c.no === +m[1]));
    }
  });
  document.addEventListener("click", e => {
    const a = e.target.closest && e.target.closest("[data-jump]");
    if (a) { e.preventDefault(); const el = document.getElementById(a.dataset.jump); if (el) el.scrollIntoView({ behavior: "smooth", block: "start" }); }
  });
  window.addEventListener("hashchange", route);
  window.addEventListener("DOMContentLoaded", () => {
    BOOK.chapters.sort((a, b) => a.no - b.no);
    route();
  });
})();
