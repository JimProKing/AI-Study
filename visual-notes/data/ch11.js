BOOK.add({
  no: 11,
  title: "이미지 분할",
  short: "이미지 분할",
  one: "픽셀마다 라벨 붙이기: 직접 만들거나 SAM을 쓰거나",
  cards: [
    {
      id: "11.1-tasks", sec: "11.1", title: "컴퓨터 비전 작업",
      fig: { h: 214, alt: "이미지 분류는 라벨, 이미지 분할은 픽셀별 영역, 객체 탐지는 사각형 상자를 낸다", f: k => {
        const scene = (x) => [
          k.box(x, 30, 200, 130, ""),
          k.circle(x + 62, 100, 32, null, { fill: "orange" }),
          k.path(`M${x + 120} 140L${x + 150} 72L${x + 180} 140Z`, { arrow: false, fill: "blue-a" }),
        ];
        return [
          ...scene(20), k.box(28, 38, 70, 24, "고양이", { fill: "yellow", size: 12 }),
          ...scene(280),
          k.raw(`<circle class="mark m-red" cx="342" cy="100" r="34"/>`), k.path("M398 142L430 70L462 142Z", { arrow: false, fill: "mint-a" }),
          ...scene(540),
          k.box(568, 66, 68, 68, "", { fill: "none", color: "red", thick: true }), k.box(656, 68, 64, 76, "", { fill: "none", color: "teal", thick: true }),
          k.ctext(120, 186, "① 이미지 분류", { bold: true, size: 14, u: true }), k.ctext(380, 186, "② 이미지 분할", { bold: true, size: 14, u: true }), k.ctext(640, 186, "③ 객체 탐지 (12장)", { bold: true, size: 14, u: true }),
          k.ctext(120, 206, "이미지 → 라벨", { size: 12, color: "muted" }), k.ctext(380, 206, "픽셀마다 → 라벨", { size: 12, color: "muted" }), k.ctext(640, 206, "물체마다 → 상자 + 라벨", { size: 12, color: "muted" }),
        ];
      } },
      pts: [
        "· 분할 유형 3가지",
        "  · ① ==시맨틱 분할==: 픽셀마다 클래스 ('고양이'), 고양이가 2마리여도 구분 X",
        "  · ② ==인스턴스 분할==: 개별 물체를 구분 (고양이 1, 고양이 2)",
        "  · ③ ==파노프틱 분할==: ① + ② 결합",
      ],
      end: "*결국, '뭐가' → '어디에 픽셀로' → '어디에 상자로'",
    },
    {
      id: "11.2-seg-model", sec: "11.2", title: "밑바닥부터 분할 모델",
      fig: { h: 222, alt: "스트라이드 합성곱으로 압축하고 전치 합성곱으로 원래 크기의 마스크로 되돌린다", f: k => {
        const enc = [[20, 140], [80, 104], [134, 72], [182, 48]], dec = [[490, 48], [538, 72], [592, 104], [652, 140]];
        const out = [];
        enc.forEach(([x, h], i) => out.push(k.box(x, 100 - h / 2, 40, h, "", { fill: i === 0 ? "orange" : "mint", r: 1 })));
        dec.forEach(([x, h], i) => out.push(k.box(x, 100 - h / 2, 40, h, "", { fill: i === 3 ? "red" : "yellow", r: 1 })));
        out.push(k.arrow(226, 100, 486, 100, { label: "작은 특성 맵", lc: "ink" }));
        out.push(k.ctext(120, 188, "Conv2D (strides=2)\n압축", { size: 12, mono: false }));
        out.push(k.ctext(590, 188, "Conv2DTranspose\n원래 크기로", { size: 12 }));
        out.push(k.ctext(40, 22, "이미지", { size: 12, color: "muted" }), k.ctext(672, 22, "분할 마스크", { size: 12, color: "muted" }));
        return out;
      } },
      pts: [
        "· 데이터: ==Oxford-IIIT Pets== (반려동물 사진 + 분할 마스크)",
        "· 인코더: 스트라이드 `Conv2D` 로 이미지를 작은 특성 맵으로 '압축'",
        "  · 분할은 위치 정보가 중요 → 최대 풀링 대신 스트라이드로 줄임",
        "· 디코더: `Conv2DTranspose` 로 다시 입력 크기만큼 키워 픽셀별 클래스 출력",
      ],
      end: "*결국, 줄였다가 다시 키워서 픽셀마다 답하기",
    },
    {
      id: "11.3-sam", sec: "11.3", title: "SAM (Segment Anything)",
      fig: { h: 226, alt: "이미지 인코더, 프롬프트 인코더, 마스크 디코더로 이뤄진 SAM", f: k => [
        k.box(20, 30, 120, 70, "", { fill: "orange" }), k.ctext(80, 116, "이미지", { size: 12, color: "muted" }),
        k.arrow(142, 65, 186, 65),
        k.box(190, 40, 160, 50, "이미지 인코더", { fill: "blue", bold: true }),
        k.box(20, 140, 120, 50, "프롬프트", { sub: "점 · 박스 · 텍스트" }),
        k.arrow(142, 165, 186, 165),
        k.box(190, 140, 160, 50, "프롬프트 인코더", { fill: "mint", bold: true }),
        k.arrow(352, 65, 416, 100), k.arrow(352, 165, 416, 128),
        k.box(420, 88, 150, 52, "마스크 디코더", { fill: "yellow", bold: true }),
        k.arrow(572, 114, 610, 114),
        k.box(614, 74, 110, 80, "", {}), k.raw(`<circle class="mark m-red" cx="669" cy="114" r="28"/>`),
        k.ctext(669, 172, "예측 마스크", { size: 12, color: "muted" }),
        k.red(420, 40, "'여기 클릭' 하면\n그 물체만 잘라 줌", { size: 17 }),
      ] },
      pts: [
        "· ==SAM==: 메타 AI가 2023년에 공개한 사전 훈련 분할 모델 (KerasHub에 있음)",
        "· 1,100만 장 이미지 + 10억 개 넘는 마스크(==SA-1B== 데이터셋)로 훈련",
        "· 구조: ① 이미지 인코더 ② 프롬프트 인코더 ③ 마스크 디코더",
        "· ==프롬프트==로 원하는 물체 지정: 포인트, 박스 등",
      ],
      end: "*결국, 직접 훈련 안 해도 '찍으면 잘라 주는' 모델",
    },
  ],
});
