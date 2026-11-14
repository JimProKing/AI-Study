BOOK.add({
  no: 9,
  title: "컴퓨터 비전을 위한 고급 딥러닝",
  short: "고급 컴퓨터 비전",
  one: "잔차 연결 + 배치 정규화 + 분리 합성곱 = 최신 컨브넷의 기본기",
  cards: [
    {
      id: "9.1-tasks", sec: "9.1", title: "컴퓨터 비전 3대 작업",
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
          k.ctext(120, 186, "① 이미지 분류", { bold: true, size: 14, u: true }), k.ctext(380, 186, "② 이미지 분할", { bold: true, size: 14, u: true }), k.ctext(640, 186, "③ 객체 탐지", { bold: true, size: 14, u: true }),
          k.ctext(120, 206, "이미지 → 라벨", { size: 12, color: "muted" }), k.ctext(380, 206, "픽셀마다 → 라벨", { size: 12, color: "muted" }), k.ctext(640, 206, "물체마다 → 상자 + 라벨", { size: 12, color: "muted" }),
        ];
      } },
      pts: [
        "· ==이미지 분류==: 이미지 하나에 라벨 하나 (또는 여러 개)",
        "· ==이미지 분할==: 픽셀 단위로 영역 나누기 (시맨틱 분할 / 인스턴스 분할)",
        "· ==객체 탐지==: 물체 위치에 사각형(바운딩 박스) + 라벨",
        "· 9.2 예제: 분할 모델 = Conv로 줄이고 `Conv2DTranspose` 로 다시 원래 크기로 키움",
      ],
      end: "*결국, '뭐가' → '어디에 픽셀로' → '어디에 상자로'",
    },
    {
      id: "9.3-residual", sec: "9.3.2", title: "잔차 연결",
      fig: { h: 220, alt: "입력 x를 블록을 건너뛰어 출력에 더하는 잔차 연결", f: k => [
        k.oval(70, 120, 40, 26, "x"),
        k.arrow(112, 120, 176, 120),
        k.box(180, 92, 120, 56, "Conv", { sub: "+ 활성화" }), k.arrow(302, 120, 336, 120),
        k.box(340, 92, 120, 56, "Conv", { sub: "+ 활성화" }), k.arrow(462, 120, 514, 120),
        k.circle(536, 120, 20, "+", { fill: "yellow", size: 22, bold: true }),
        k.arrow(558, 120, 616, 120), k.oval(668, 120, 50, 26, "출력"),
        k.path("M140 120V40H536V98", { color: "red", thick: true }),
        k.red(240, 30, "지름길: x를 그대로 더함", { size: 18 }),
        k.red(160, 196, "출력 = block(x) + x  → 정보가 손실 없이 흐름", { size: 18 }),
      ] },
      pts: [
        "· 층이 깊어지면 ==그레이디언트 소실==: 앞쪽 층까지 학습 신호가 거의 안 감",
        "· ==잔차 연결==: 블록의 입력을 출력에 더함 → 정보 · 그레이디언트의 고속도로",
        "· 크기가 다르면 `Conv2D(필터 수, 1)` (1×1 합성곱)로 맞춰서 더함",
      ],
      end: "*결국, 깊게 쌓으려면 지름길을 깔아라",
    },
    {
      id: "9.3-batchnorm", sec: "9.3.3", title: "배치 정규화",
      fig: { h: 214, alt: "들쭉날쭉한 분포를 평균 0, 분산 1로 맞추는 배치 정규화", f: k => [
        k.plot(30, 40, 260, 130, { xl: "값", curves: [{ f: t => 0.85 * Math.exp(-((t - 0.72) ** 2) / 0.006), color: "muted", thick: true }, { f: t => 0.5 * Math.exp(-((t - 0.3) ** 2) / 0.03), color: "muted", dash: true }] }),
        k.ctext(160, 200, "층마다 들쭉날쭉한 분포", { size: 12, color: "muted" }),
        k.arrow(310, 105, 420, 105, { label: "BatchNormalization" }),
        k.plot(440, 40, 260, 130, { xl: "값", curves: [{ f: t => 0.8 * Math.exp(-((t - 0.5) ** 2) / 0.02), color: "teal", thick: true }], vlines: [{ t: 0.5, label: "0", color: "red" }] }),
        k.red(470, 22, "평균 0 · 분산 1로 맞춤", { size: 18 }),
      ] },
      pts: [
        "· ==배치 정규화==: 훈련 중 배치의 평균 · 분산으로 층 출력을 정규화",
        "· 추론 때는 훈련 중 모아 둔 이동 평균을 사용",
        "· 효과: 그레이디언트 전파가 쉬워짐 → 더 깊은 모델 훈련 가능",
        "· 요령: Conv/Dense 층은 `use_bias=False`, 활성화는 BN ==뒤에== 적용",
        "· 미세 조정할 때 BN 층은 동결해 두는 게 좋음",
      ],
      end: "*결국, 층 사이 값을 늘 비슷한 범위로 정돈",
    },
    {
      id: "9.3-separable", sec: "9.3.4", title: "깊이별 분리 합성곱",
      fig: { h: 222, alt: "채널마다 따로 공간 합성곱을 한 뒤 1×1 합성곱으로 채널을 섞는다", f: k => [
        k.text(20, 22, "일반 Conv: 공간 + 채널을 한꺼번에", { size: 13, color: "muted" }),
        k.cube(30, 80, 3, 3, 3, { cw: 18, fill: "orange", backfill: "orange" }),
        k.arrow(110, 100, 150, 100, { label: "채널 분리" }),
        ...[0, 1, 2].map(i => k.grid(160 + i * 76, 70, 3, 3, null, { cw: 18, fill: "orange" })),
        ...[0, 1, 2].map(i => k.arrow(187 + i * 76, 128, 187 + i * 76, 150)),
        ...[0, 1, 2].map(i => k.box(166 + i * 76, 152, 42, 30, "3×3", { size: 12, mono: true, fill: "mint" })),
        k.ctext(264, 204, "① 깊이별: 채널마다 따로 공간 합성곱", { size: 12 }),
        k.arrow(394, 167, 450, 167),
        k.box(454, 142, 120, 50, "1×1 Conv", { mono: true, sub: "② 점별: 채널 섞기", fill: "yellow" }),
        k.arrow(576, 167, 616, 167), k.oval(668, 167, 50, 26, "출력"),
        k.red(454, 70, "파라미터 · 연산 훨씬 적음\n→ 작은 데이터에도 강함", { size: 17 }),
      ] },
      pts: [
        "· 가정: ==공간 패턴==과 ==채널 간 관계==는 서로 독립 → 따로 배워도 됨",
        "· `SeparableConv2D` 층 하나로 사용",
        "· 9.3.5 ==Xception== 스타일: 블록 반복 + 잔차 연결 + 배치 정규화 + 분리 합성곱",
      ],
      end: "*결국, 한 번에 할 일을 두 번에 나눠서 싸고 빠르게",
    },
    {
      id: "9.4-interpret", sec: "9.4", title: "컨브넷이 본 것 해석하기",
      fig: { h: 214, alt: "중간 활성화, 필터 패턴, 클래스 활성화 히트맵 세 가지 시각화", f: k => {
        const heat = [];
        for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) heat.push(Math.max(0, 1 - Math.hypot(i - 2.2, j - 3.4) / 3));
        const stripes = [];
        for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) stripes.push((i + j) % 3 === 0 ? 1 : 0.1);
        return [
          ...[0, 1, 2, 3].map(i => k.grid(30 + (i % 2) * 78, 30 + Math.floor(i / 2) * 78, 5, 5, null, { cw: 14, thin: true, heat: Array.from({ length: 25 }, (_, n) => ((n * (i + 3)) % 7) / 7) })),
          k.ctext(110, 196, "① 중간 활성화", { bold: true, size: 14, u: true }),
          k.grid(300, 40, 6, 6, null, { cw: 22, thin: true, heat: stripes, heatc: "orange" }),
          k.ctext(366, 196, "② 필터 시각화", { bold: true, size: 14, u: true }),
          k.box(540, 36, 150, 140, ""),
          k.grid(540, 36, 6, 6, null, { cw: 25, thin: true, heat, heatc: "red" }),
          k.circle(625, 96, 26, null, { fill: "none" }),
          k.ctext(615, 196, "③ 클래스 활성화 맵", { bold: true, size: 14, u: true }),
        ];
      } },
      pts: [
        "· ① ==중간 활성화==: 층마다 특성 맵을 이미지로 → 아래층은 모서리, 위층일수록 추상적",
        "· ② ==필터 시각화==: 입력 공간에 경사 상승법 → 필터가 가장 좋아하는 패턴",
        "· ③ ==클래스 활성화 맵==(Grad-CAM): 이 클래스로 판단한 근거가 이미지의 어디인지 히트맵",
        "· 컨브넷은 블랙박스가 아님!",
      ],
      end: "*결국, 모델이 '어디를 보고' 판단했는지 눈으로 확인 가능",
    },
  ],
});
