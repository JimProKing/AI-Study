BOOK.add({
  no: 8,
  title: "이미지 분류",
  short: "이미지 분류",
  one: "작은 창으로 훑으며 지역 패턴을 층층이 쌓는다 = 컨브넷",
  cards: [
    {
      id: "8.1-conv", sec: "8.1.1", title: "합성곱 연산",
      fig: { h: 240, alt: "5×5 입력 위를 3×3 필터가 움직이며 점곱 합을 구해 3×3 특성 맵을 만든다", f: k => [
        k.text(20, 24, "입력 (5×5)", { size: 12, color: "muted" }),
        k.grid(20, 40, 5, 5, [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1], { cw: 30, hi: [0, 1, 2, 5, 6, 7, 10, 11, 12] }),
        k.ctext(196, 115, "*", { size: 24, bold: true }),
        k.text(220, 56, "필터 (3×3)", { size: 12, color: "muted" }),
        k.grid(220, 70, 3, 3, [1, 0, 1, 0, 1, 0, 1, 0, 1], { cw: 30, fill: "mint" }),
        k.arrow(320, 115, 372, 115, { label: "곱해서 합" }),
        k.text(384, 56, "특성 맵 (3×3)", { size: 12, color: "muted" }),
        k.grid(384, 70, 3, 3, [4, 2, 4, 2, 4, 2, 3, 2, 4], { cw: 30, hi: [0], hic: "red" }),
        k.box(500, 26, 240, 58, "① 이동 불변성", { sub: "한 곳에서 배운 패턴을 어디서든", fill: "yellow" }),
        k.box(500, 96, 240, 58, "② 공간적 계층 구조", { sub: "작은 패턴 → 큰 패턴", fill: "mint" }),
        k.red(500, 196, "Dense = 전역 패턴\nConv = 지역 패턴", { size: 18 }),
        k.red(20, 222, "창을 한 칸씩 밀면서 반복", { size: 17 }),
      ] },
      pts: [
        "· 핵심 파라미터 ① ==패치 크기== (보통 3×3) ② ==출력 깊이== = 필터 개수",
        "· ==패딩==: 가장자리에 0을 둘러 출력 크기 유지 (`padding=\"same\"`)",
        "· ==스트라이드==: 창을 몇 칸씩 건너뛸지",
        "· 출력 = ==특성 맵== (높이, 너비, 필터 수)",
      ],
      end: "*결국, 작은 창으로 훑으며 패턴 찾기",
    },
    {
      id: "8.1-maxpool", sec: "8.1.2", title: "최대 풀링",
      fig: { h: 204, alt: "4×4 특성 맵을 2×2 칸마다 최댓값만 남겨 2×2로 줄인다", f: k => [
        k.grid(60, 30, 2, 2, [1, 3, 5, 6], { cw: 36, fill: "yellow" }),
        k.grid(132, 30, 2, 2, [2, 4, 1, 2], { cw: 36, fill: "mint" }),
        k.grid(60, 102, 2, 2, [7, 2, 1, 8], { cw: 36, fill: "mint" }),
        k.grid(132, 102, 2, 2, [9, 1, 3, 4], { cw: 36, fill: "yellow" }),
        k.ctext(132, 192, "(4, 4)", { mono: true, size: 12, color: "muted" }),
        k.arrow(220, 102, 300, 102, { label: "2×2 최댓값" }),
        k.box(320, 66, 36, 36, "6", { fill: "yellow", r: 0, bold: true }), k.box(356, 66, 36, 36, "4", { fill: "mint", r: 0, bold: true }),
        k.box(320, 102, 36, 36, "8", { fill: "mint", r: 0, bold: true }), k.box(356, 102, 36, 36, "9", { fill: "yellow", r: 0, bold: true }),
        k.ctext(356, 156, "(2, 2)", { mono: true, size: 12, color: "muted" }),
        k.red(450, 50, "크기 절반으로!", { size: 20 }),
        k.text(450, 92, "왜?", { bold: true, size: 14 }),
        k.text(450, 122, "① 처리할 원소 수 ↓ (과대적합 ↓)", { size: 13 }),
        k.text(450, 150, "② 다음 층이 더 넓은 영역을 봄", { size: 13 }),
      ] },
      pts: [
        "· 보통 2×2 창, 스트라이드 2 → 높이 · 너비 절반",
        "· 평균 풀링보다 ==최대 풀링==이 보통 더 잘 됨 (특성이 '있는지'가 중요)",
        "· 풀링 없이 Conv만 쌓으면 마지막 층도 입력의 작은 부분만 봄",
      ],
      end: "*결국, 풀링 = 요약해서 줄이기",
    },
    {
      id: "8.2-convnet", sec: "8.2", title: "컨브넷 구조 (강아지 vs 고양이)",
      fig: { h: 228, alt: "특성 맵의 크기는 줄고 깊이는 늘어나다가 Flatten과 Dense로 분류한다", f: k => {
        const blocks = [["180×180×3", 12, 150], ["89×89×32", 18, 120], ["43×43×64", 24, 96], ["20×20×128", 30, 72], ["9×9×256", 36, 52], ["7×7×256", 36, 40]];
        const xs = [24, 96, 166, 234, 300, 362];
        const out = [k.red(96, 22, "크기는 ↓   깊이(필터 수)는 ↑", { size: 18 })];
        blocks.forEach(([l, w, h], i) => {
          out.push(k.box(xs[i], 115 - h / 2, w, h, "", { fill: i === 0 ? "orange" : "mint", r: 1 }));
          out.push(k.ctext(xs[i] + w / 2, i % 2 ? 214 : 198, l, { size: 11, mono: true, color: "muted" }));
          if (i < blocks.length - 1) out.push(k.arrow(xs[i] + w + 4, 115, xs[i + 1] - 4, 115));
        });
        out.push(k.arrow(402, 115, 440, 115), k.box(444, 95, 90, 40, "Flatten", { mono: true, size: 13 }));
        out.push(k.arrow(536, 115, 566, 115), k.box(570, 89, 120, 52, "Dense(1)", { sub: "sigmoid", mono: true, fill: "yellow" }));
        out.push(k.red(574, 172, "개 vs 고양이", { size: 18 }));
        return out;
      } },
      pts: [
        "· 데이터: 강아지 vs 고양이 사진, 훈련 ==2,000장== / 검증 1,000 / 테스트 2,000",
        "· 정확도: 밑바닥부터 약 70% → !!데이터 증강 80~85%!! → !!사전 훈련 특성 추출 97.5%!! (8.3)",
        "· `image_dataset_from_directory` 로 폴더에서 바로 데이터셋 만들기",
        "· 입력은 `Rescaling(1./255)` 로 0~1 스케일",
        "· 특성 맵: 깊이 32 → 256 증가, 크기 180 → 7 감소 (컨브넷 공통 패턴)",
      ],
      end: "*결국, Conv → Pool 반복 → Flatten → Dense",
    },
    {
      id: "8.2-augment", sec: "8.2.5", title: "데이터 증강",
      fig: { h: 212, alt: "고양이 사진 한 장을 뒤집기, 회전, 확대로 여러 장처럼 만든다", f: k => {
        const cat = `<circle class="sh f-orange" cx="0" cy="4" r="24"/><path class="sh f-orange" d="M-20 -8L-16 -30L-4 -18ZM20 -8L16 -30L4 -18Z"/><circle class="dot d-ink" cx="-8" cy="2" r="3"/><circle class="dot d-ink" cx="8" cy="2" r="3"/><path class="ln s-ink" d="M-4 12Q0 16 4 12"/><circle class="dot d-red" cx="14" cy="-16" r="3.5"/>`;
        const g = (cx, cy, t) => k.raw(`<g transform="translate(${cx} ${cy}) ${t}">${cat}</g>`);
        const out = [k.box(20, 50, 100, 100, ""), g(70, 102, ""), k.ctext(70, 170, "원본", { size: 13, bold: true })];
        out.push(k.arrow(124, 100, 196, 100, { label: "증강" }));
        [["RandomFlip", "scale(-1 1)"], ["RandomRotation", "rotate(18)"], ["RandomZoom", "scale(1.3)"], ["조합", "scale(-1 1) rotate(-14)"]].forEach(([l, t], i) => {
          const x = 200 + i * 135;
          out.push(k.box(x, 50, 110, 100, "", { fill: "mint" }), g(x + 55, 102, t), k.ctext(x + 55, 170, l, { size: 12, mono: i < 3 }));
        });
        out.push(k.red(200, 200, "매 에포크마다 조금씩 다른 사진 → 외우기 어려움", { size: 17 }));
        return out;
      } },
      pts: [
        "· 작은 데이터셋 → 과대적합이 최대 문제",
        "· ==데이터 증강==: 그럴듯한 랜덤 변환으로 샘플을 늘림",
        "· 케라스 층으로 모델 앞에 붙임: `RandomFlip`, `RandomRotation`, `RandomZoom`",
        "· 추론(predict) 때는 자동으로 꺼짐",
        "· 드롭아웃과 같이 쓰면 효과 ↑",
      ],
      end: "*결국, 같은 사진을 다르게 보여줘서 일반화 ↑",
    },
    {
      id: "8.3-pretrained", sec: "8.3", title: "사전 훈련된 모델 활용",
      fig: { h: 228, alt: "특성 추출은 합성곱 기반 전체를 동결하고, 미세 튜닝은 상위 층 일부를 풀어 다시 학습한다", f: k => [
        k.text(20, 45, "① 특성 추출", { bold: true, size: 14 }),
        k.box(150, 20, 300, 50, "Xception 합성곱 기반", { fill: "blue", sub: "전체 동결" }),
        k.arrow(452, 45, 486, 45), k.box(490, 20, 180, 50, "새 분류기", { sub: "Dense", fill: "yellow" }),
        k.text(150, 88, "ImageNet으로 미리 훈련된 가중치", { size: 12, color: "muted" }),
        k.text(20, 135, "② 미세 튜닝", { bold: true, size: 14 }),
        k.box(150, 110, 200, 50, "하위 층", { fill: "blue", sub: "동결" }),
        k.box(350, 110, 100, 50, "상위 층", { fill: "mint", sub: "조금 학습" }),
        k.arrow(452, 135, 486, 135), k.box(490, 110, 180, 50, "새 분류기", { fill: "yellow" }),
        k.red(150, 196, "순서: 분류기 먼저 훈련 → 그다음 상위 층 풀기 (작은 학습률)", { size: 16 }),
      ] },
      pts: [
        "· ==사전 훈련 모델==: ImageNet(140만 장, 1,000 클래스)으로 이미 훈련된 모델",
        "· 하위 층 = 일반적인 특성 (모서리 · 색 · 질감) → 재사용 가치 큼",
        "· 상위 층 = 원래 문제에 특화된 특성",
        "· `conv_base.trainable = False` 로 동결",
        "· 미세 튜닝은 ==분류기를 먼저 훈련한 뒤==에: 안 그러면 큰 오차 신호가 사전 학습 표현을 망침",
      ],
      end: "*결국, 남이 잘 배운 눈을 빌려 쓰기",
    },
  ],
});
