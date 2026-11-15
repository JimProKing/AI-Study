/* 카드에 붙는 인터랙티브 블록. 카드 데이터에 widget: "이름" 을 쓰면 app.js 가 BOOK.widgets[이름](el) 을 부른다. */
(function () {
  const BOOK = window.BOOK;
  BOOK.widgets = BOOK.widgets || {};

  /* ---------- binary_crossentropy (4.1 영화 리뷰) ---------- */
  BOOK.widgets.bce = function (el) {
    const PW = 460, PH = 230, L = 46, R = 16, T = 14, B = 34, YMAX = 5;
    const X = p => L + p * (PW - L - R), Y = v => T + (1 - Math.min(v, YMAX) / YMAX) * (PH - T - B);
    const curve = f => { let d = ""; for (let i = 1; i <= 199; i++) { const p = i / 200, v = f(p); d += (d ? "L" : "M") + X(p).toFixed(1) + " " + Y(v).toFixed(1); } return d; };
    const ticksY = [0, 1, 2, 3, 4, 5].map(v => `<line x1="${L - 4}" y1="${Y(v)}" x2="${PW - R}" y2="${Y(v)}" class="g"/><text x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`).join("");
    const ticksX = [0, 0.25, 0.5, 0.75, 1].map(p => `<text x="${X(p)}" y="${PH - B + 18}" text-anchor="middle">${p}</text>`).join("");

    el.innerHTML = `
      <div class="w-head"><span class="w-tag">직접 돌려보기</span><b>binary_crossentropy</b></div>
      <div class="w-math">
        <div class="w-eq-label">샘플 하나</div>
        <math display="block"><mi>ℒ</mi><mo stretchy="false">(</mo><mi>y</mi><mo>,</mo><mi>p</mi><mo stretchy="false">)</mo><mo>=</mo><mo>−</mo><mo>[</mo>
          <mi>y</mi><mo>·</mo><mi>log</mi><mo>&#x2061;</mo><mi>p</mi><mo>+</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>y</mi><mo stretchy="false">)</mo><mo>·</mo><mi>log</mi><mo>&#x2061;</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>p</mi><mo stretchy="false">)</mo><mo>]</mo></math>
        <div class="w-eq-label">배치 (케라스가 실제로 보는 값)</div>
        <math display="block"><mi>ℒ</mi><mo>=</mo><mo>−</mo><mfrac><mn>1</mn><mi>N</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mo>[</mo>
          <msub><mi>y</mi><mi>i</mi></msub><mi>log</mi><mo>&#x2061;</mo><msub><mi>p</mi><mi>i</mi></msub><mo>+</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><msub><mi>y</mi><mi>i</mi></msub><mo stretchy="false">)</mo><mi>log</mi><mo>&#x2061;</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><msub><mi>p</mi><mi>i</mi></msub><mo stretchy="false">)</mo><mo>]</mo></math>
        <ul class="w-legend">
          <li><i>y</i> = 정답 레이블 (긍정 1 · 부정 0)</li>
          <li><i>p</i> = <code>Dense(1, sigmoid)</code> 출력 = 긍정일 확률</li>
          <li>log = 자연로그 ln</li>
        </ul>
      </div>
      <div class="w-ctrl">
        <div class="w-seg" role="radiogroup" aria-label="정답 레이블 y">
          <button type="button" data-y="1" role="radio">긍정 리뷰 · y = 1</button>
          <button type="button" data-y="0" role="radio">부정 리뷰 · y = 0</button>
        </div>
        <label class="w-slider">모델 예측 <i>p</i> = <output class="w-p"></output>
          <input type="range" min="0.01" max="0.99" step="0.01" value="0.73" aria-label="모델이 낸 긍정 확률 p">
        </label>
      </div>
      <div class="w-sub" aria-live="polite"></div>
      <svg class="w-plot" viewBox="0 0 ${PW} ${PH}" role="img" aria-label="예측 확률 p에 따른 손실 곡선. y=1이면 −log p, y=0이면 −log(1−p)">
        ${ticksY}${ticksX}
        <line x1="${L}" y1="${PH - B}" x2="${PW - R}" y2="${PH - B}" class="ax"/><line x1="${L}" y1="${T}" x2="${L}" y2="${PH - B}" class="ax"/>
        <text x="${(L + PW - R) / 2}" y="${PH - 2}" text-anchor="middle" class="lab">예측 p (긍정 확률)</text>
        <text x="12" y="${(T + PH - B) / 2}" text-anchor="middle" class="lab" transform="rotate(-90 12 ${(T + PH - B) / 2})">손실</text>
        <path class="c c1" d="${curve(p => -Math.log(p))}"/>
        <path class="c c0" d="${curve(p => -Math.log(1 - p))}"/>
        <text class="cl cl1" x="${X(0.06)}" y="${Y(-Math.log(0.06)) - 6}">−log p (y=1)</text>
        <text class="cl cl0" x="${X(0.94)}" y="${Y(-Math.log(0.06)) - 6}" text-anchor="end">−log(1−p) (y=0)</text>
        <line class="vl"/><circle class="pt" r="6"/>
        <text class="pv"></text>
      </svg>
      <p class="w-verdict"></p>`;

    const st = { y: 1, p: 0.73 };
    const q = s => el.querySelector(s);
    const f2 = v => v.toFixed(2), f3 = v => v.toFixed(3);

    function draw() {
      const { y, p } = st, loss = -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
      el.querySelectorAll(".w-seg button").forEach(b => { const on = +b.dataset.y === y; b.classList.toggle("on", on); b.setAttribute("aria-checked", on); });
      el.classList.toggle("y0", y === 0);
      q(".w-p").textContent = f2(p);
      const a = y === 1 ? "" : " dead", b = y === 0 ? "" : " dead";
      q(".w-sub").innerHTML =
        `<span class="w-eq">ℒ = −[ <span class="t${a}">${y}·log(${f2(p)})</span> + <span class="t${b}">${1 - y}·log(${f2(1 - p)})</span> ]</span>` +
        `<span class="w-eq">= −log(${f2(y ? p : 1 - p)}) = <b>${f3(loss)}</b></span>` +
        `<span class="w-note">y가 0 아니면 1이라, 두 항 중 <b>하나만 살아남는다</b> → 정답 쪽에 준 확률의 −log</span>`;
      const cx = X(p), cy = Y(loss);
      q(".vl").setAttribute("x1", cx); q(".vl").setAttribute("x2", cx); q(".vl").setAttribute("y1", Y(0)); q(".vl").setAttribute("y2", cy);
      q(".pt").setAttribute("cx", cx); q(".pt").setAttribute("cy", cy);
      const pv = q(".pv"); pv.textContent = f3(loss); pv.setAttribute("x", cx + (p > 0.75 ? -10 : 10)); pv.setAttribute("y", Math.max(cy - 10, 26)); pv.setAttribute("text-anchor", p > 0.75 ? "end" : "start");
      const right = y ? p : 1 - p;
      q(".w-verdict").textContent =
        right >= 0.9 ? "정답 쪽에 확신 → 손실 거의 0. 고칠 게 별로 없음" :
        right >= 0.5 ? "맞히긴 했는데 자신 없음 → 손실 조금. 옵티마이저가 살짝 밀어 줌" :
        right >= 0.1 ? "틀림 → 손실이 커지기 시작" :
        "확신하면서 틀림 → 손실 폭발! (−log가 0 근처에서 ∞로) 그레이디언트도 크게 나옴";
    }
    el.addEventListener("click", e => { const b = e.target.closest("[data-y]"); if (b) { st.y = +b.dataset.y; draw(); } });
    q('input[type="range"]').addEventListener("input", e => { st.p = +e.target.value; draw(); });
    draw();
  };
})();
