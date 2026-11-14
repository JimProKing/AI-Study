BOOK.add({
  no: 2,
  title: "신경망의 수학적 구성 요소",
  short: "수학적 구성 요소",
  one: "텐서 + 텐서 연산 + 경사 하강법 = 신경망의 전부",
  cards: [
    {
      id: "2.1-first-network", sec: "2.1", title: "신경망과의 첫 만남 (MNIST)",
      fig: { h: 214, alt: "28×28 손글씨 이미지를 784 벡터로 펴서 Dense 512, Dense 10 softmax를 지나 숫자별 확률을 낸다", f: k => [
        k.grid(20, 50, 6, 6, null, { cw: 14, thin: true, hi: [1, 2, 3, 4, 10, 15, 21, 26, 32], hic: "ink" }),
        k.ctext(62, 152, "28×28 흑백", { size: 12, color: "muted" }),
        k.arrow(112, 92, 150, 92, { label: "펴기" }),
        k.box(154, 62, 96, 60, "벡터", { sub: "(784,)", submono: true }),
        k.arrow(252, 92, 288, 92),
        k.box(292, 52, 120, 80, "Dense 512", { sub: "relu" }),
        k.arrow(414, 92, 450, 92),
        k.box(454, 52, 120, 80, "Dense 10", { sub: "softmax", fill: "mint" }),
        k.arrow(576, 92, 606, 92),
        k.bars(612, 40, 128, 80, [.01, .01, .02, .01, .005, .01, .005, .9, .02, .02], { hi: [7], labels: "0123456789".split(""), gap: 3 }),
        k.red(612, 158, "\"7\"일 확률 0.9!", { size: 18 }),
        k.red(154, 192, "compile 3요소 = 옵티마이저 · 손실 · 지표", { size: 18 }),
      ] },
      pts: [
        "· ==MNIST==: 28×28 손글씨 숫자 → 0~9 클래스 10개 (훈련 6만 장, 테스트 1만 장)",
        "· 컴파일 3요소 ① ==옵티마이저== `rmsprop` ② ==손실 함수== `sparse_categorical_crossentropy` ③ ==지표== `accuracy`",
        "· `model.fit(train_images, train_labels, epochs=5, batch_size=128)`",
        "· 훈련 정확도 98.9% > 테스트 97.8% → 이 차이가 ==과대적합== (5장에서 자세히)",
      ],
      end: "*결국, 층 2개 + compile + fit = 딥러닝 기본형",
    },
    {
      id: "2.2-tensor", sec: "2.2", title: "텐서",
      mine: { img: "my-notes/tensor.jpg", date: "2026-11-14" },
      fig: { h: 214, alt: "스칼라, 벡터, 매트릭스, 텐서로 축이 하나씩 늘어난다", f: k => [
        k.cells(40, 60, [1], { cw: 44 }),
        k.arrow(100, 82, 150, 82),
        k.cells(160, 60, [1, 2, 3, 4], { cw: 40 }),
        k.arrow(330, 82, 380, 82),
        k.grid(395, 46, 2, 2, [1, 2, 3, 4], { cw: 36 }),
        k.arrow(482, 82, 532, 82),
        k.cube(560, 62, 2, 2, 3, { cw: 30, vals: [1, 2, 3, 4] }),
        k.ctext(62, 146, "스칼라", { size: 16, u: true }), k.ctext(240, 146, "벡터", { size: 16, u: true }),
        k.ctext(431, 146, "매트릭스", { size: 16, u: true }), k.ctext(604, 146, "텐서", { size: 16, u: true }),
        k.ctext(62, 170, "랭크-0", { size: 12, color: "muted" }), k.ctext(240, 170, "랭크-1 · 축 1개", { size: 12, color: "muted" }),
        k.ctext(431, 170, "랭크-2 · 축 2개", { size: 12, color: "muted" }), k.ctext(604, 170, "랭크-3 이상", { size: 12, color: "muted" }),
        k.red(40, 200, "+책  핵심 속성 3개 = ndim(랭크) · shape(크기) · dtype(자료형)", { size: 17 }),
      ] },
      pts: [
        "· 관점 ① ==딥러닝== : 데이터 담는 그릇",
        "  ② ==물리학·수학== : 좌표계와 무관한 법칙",
        "· ==dtype== : 텐서 내부 요소 자료형",
        "  · 변경법 ① `tensor.to(torch.float64)` ② `tensor.float()`",
        "  · !!+책 (넘파이 버전)!! `x.astype(\"float64\")`",
        "· !!+책!! ==배치 축== : 첫 번째 축 = 샘플 축. 예: `train_images[:128]`",
      ],
      end: "*2장은 넘파이로 설명해요. 파이토치는 3장에 나오니 내 torch 코드 그대로 OK",
    },
    {
      id: "2.2-real-tensors", sec: "2.2.8", title: "실제 데이터의 텐서 크기",
      fig: { h: 244, alt: "벡터, 시계열, 이미지, 비디오 데이터의 랭크와 shape", f: k => {
        const rows = [
          ["벡터 데이터", "랭크-2", "(samples, features)", "예) 고객 10만 명 × 3개 특성"],
          ["시계열·시퀀스", "랭크-3", "(samples, timesteps, features)", "예) (250, 390, 3) 주가"],
          ["이미지", "랭크-4", "(samples, height, width, channels)", "예) (128, 256, 256, 3)"],
          ["비디오", "랭크-5", "(samples, frames, height, width, channels)", "예) (4, 240, 144, 256, 3)"],
        ];
        const out = [];
        rows.forEach(([n, r, sh, ex], i) => {
          const y = 20 + i * 52;
          out.push(k.text(20, y + 15, n, { bold: true, size: 14 }));
          out.push(k.box(146, y, 70, 30, r, { fill: "yellow", size: 13 }));
          out.push(k.hl(240, y + 4, 56, 22, "mint"));
          out.push(k.text(232, y + 15, sh, { mono: true, size: 13 }));
          out.push(k.text(566, y + 15, ex, { size: 12, color: "muted" }));
        });
        out.push(k.red(232, 230, "↑ 첫 번째 축 = 샘플(배치) 축", { size: 18 }));
        return out;
      } },
      pts: [
        "· 첫 번째 축 = ==샘플 축== (배치 축)",
        "· ==배치==: 데이터를 한 번에 다 X, 작은 묶음으로 → `batch = train_images[128 * n:128 * (n + 1)]`",
        "· 이미지 채널: 컬러 3 / 흑백 1",
        "· 대부분 도구는 ==채널 마지막==(channels-last), !!파이토치 기본은 채널 우선!!",
      ],
      end: "*결국, shape만 봐도 무슨 데이터인지 보인다",
    },
    {
      id: "2.3-tensor-ops", sec: "2.3", title: "텐서 연산 3종",
      fig: { h: 272, alt: "원소별 연산 relu, 브로드캐스팅, 점곱의 shape 규칙", f: k => [
        k.text(20, 22, "① 원소별 연산", { bold: true, size: 14 }),
        k.cells(20, 46, [-2, 3, -1, 5], { cw: 40 }),
        k.arrow(100, 90, 100, 120, { label: "relu", lx: 26, ly: 0, lc: "ink" }),
        k.cells(20, 126, [0, 3, 0, 5], { cw: 40, hi: [0, 2], hic: "red" }),
        k.red(20, 192, "음수 → 0", { size: 17 }),

        k.text(250, 22, "② 브로드캐스팅", { bold: true, size: 14 }),
        k.grid(250, 46, 3, 3, null, { cw: 30 }),
        k.ctext(362, 92, "+", { size: 20, bold: true }),
        k.cells(380, 46, ["b₁", "b₂", "b₃"], { cw: 30 }),
        k.grid(380, 76, 2, 3, ["b₁", "b₂", "b₃", "b₁", "b₂", "b₃"], { cw: 30, fill: "mint", tc: "muted" }),
        k.ctext(295, 152, "(3, 3)", { mono: true, size: 12, color: "muted" }), k.ctext(425, 152, "(3,) → (3, 3)", { mono: true, size: 12, color: "muted" }),
        k.red(250, 192, "작은 쪽을 반복해서\n크기 맞춤", { size: 17 }),

        k.text(510, 22, "③ 점곱 (dot)", { bold: true, size: 14 }),
        k.grid(510, 56, 3, 2, null, { cw: 22, hi: [0, 1] }),
        k.ctext(566, 89, "·", { size: 24, bold: true }),
        k.grid(578, 46, 2, 3, null, { cw: 22, hi: [0, 3] }),
        k.ctext(662, 89, "=", { size: 18, bold: true }),
        k.grid(680, 56, 3, 3, null, { cw: 20, hi: [0], hic: "red" }),
        k.ctext(532, 140, "(a, b)", { mono: true, size: 12, color: "muted" }),
        k.ctext(611, 104, "(b, c)", { mono: true, size: 12, color: "muted" }),
        k.ctext(710, 140, "(a, c)", { mono: true, size: 12, color: "muted" }),
        k.red(510, 192, "맞닿는 축(b) 크기가\n같아야 함", { size: 17 }),

        k.box(140, 226, 480, 36, "Dense 층 = relu( dot(입력, W) + b )", { fill: "yellow", mono: true, size: 14 }),
      ] },
      pts: [
        "· ==원소별 연산==: 각 원소에 독립 적용 (relu, +, ×) → 병렬 계산 쉬움",
        "· ==브로드캐스팅==: 크기 다른 텐서끼리 연산할 때 작은 텐서를 ==가상으로 반복==",
        "· ==점곱==(텐서 곱셈) `np.dot(x, y)` : (a, b) · (b, c) → (a, c)",
        "· ==크기 변환==(reshape): 원소 수 그대로, 행·열만 재배열 / 전치 `np.transpose(x)`",
      ],
      end: "*결국, 신경망 전체가 이 몇 가지 연산의 조합",
    },
    {
      id: "2.3-geometry", sec: "2.3.6", title: "딥러닝의 기하학적 해석",
      fig: { h: 212, alt: "구겨진 종이 공을 층마다 조금씩 펴서 두 클래스를 선 하나로 나눈다", f: k => [
        k.path("M70 80L95 55L130 62L160 50L172 85L158 112L175 140L140 160L110 150L80 162L58 130L66 105Z", { arrow: false, fill: "mint-a" }),
        k.path("M110 100L130 62L160 50L158 112L120 110Z", { arrow: false, fill: "orange-a" }),
        k.path("M95 55L110 100L80 162M160 50L120 110L175 140M66 105L120 110", { arrow: false, color: "muted" }),
        k.arrow(190, 105, 230, 105),
        k.box(234, 80, 70, 50, "층 1"), k.arrow(306, 105, 330, 105),
        k.box(334, 80, 70, 50, "층 2"), k.arrow(406, 105, 430, 105),
        k.box(434, 80, 70, 50, "층 3"), k.arrow(506, 105, 540, 105),
        k.path("M556 70L646 70L626 140L536 140Z", { arrow: false, fill: "mint-a" }),
        k.path("M646 70L736 70L716 140L626 140Z", { arrow: false, fill: "orange-a" }),
        k.line(646, 64, 626, 146, { color: "red", dash: true }),
        k.ctext(636, 162, "펴진 종이: 선 하나로 분리", { size: 12, color: "muted" }),
        k.red(40, 196, "구겨진 종이 공 = 얽힌 데이터", { size: 17 }),
        k.red(330, 160, "층마다 조금씩 펴기", { size: 17 }),
      ] },
      pts: [
        "· 텐서 연산 = 공간에서의 ==기하학적 변환== (이동, 회전, 크기 조절, 기울이기)",
        "· 아핀 변환(`W·x + b`)만 여러 번 쌓으면 결국 아핀 변환 하나 → 그래서 ==relu 같은 활성화 함수==가 필요",
        "· 딥러닝 = 복잡하게 얽힌 데이터를 깔끔하게 ==펴는 변환==을 찾는 것",
      ],
      end: "*결국, 층 하나 = 종이 공 한 번 펴기",
    },
    {
      id: "2.4-gradient-descent", sec: "2.4", title: "경사 하강법",
      fig: { h: 252, alt: "손실 곡선에서 그레이디언트 반대 방향으로 내려가는 모습과 미니 배치 SGD 4단계", f: k => {
        const X = t => 40 + t * 330, f = t => 0.08 + 3.2 * (t - 0.62) ** 2, Y = t => 214 - 190 * f(t);
        const ts = [0.15, 0.3, 0.42, 0.52, 0.6];
        const out = [k.plot(40, 24, 330, 190, { xl: "가중치 W", yl: "손실", curves: [{ f, from: 0.1, to: 1 }] })];
        for (let i = 0; i < ts.length - 1; i++) out.push(k.arrow(X(ts[i]) + 3, Y(ts[i]) + 3, X(ts[i + 1]) - 3, Y(ts[i + 1]) - 3, { color: "red" }));
        ts.forEach((t, i) => out.push(k.dot(X(t), Y(t), { r: 5, color: i === ts.length - 1 ? "red" : "ink" })));
        out.push(k.red(214, 54, "기울기 반대로\n한 걸음씩", { size: 17 }));
        const steps = ["① 배치 뽑기 (x, y)", "② 모델 실행 → y_pred", "③ 손실 계산 (y_pred vs y)", "④ W -= 학습률 × 그레이디언트"];
        steps.forEach((s, i) => {
          out.push(k.box(420, 24 + i * 46, 316, 34, s, { size: 13, fill: i === 3 ? "yellow" : "paper" }));
          if (i < 3) out.push(k.arrow(578, 60 + i * 46, 578, 68 + i * 46));
        });
        out.push(k.path("M736 179H752V41H740", { color: "teal" }));
        out.push(k.red(430, 226, "이걸 반복 = 미니 배치 SGD", { size: 18 }));
        return out;
      } },
      pts: [
        "· ==도함수==: 입력을 살짝 바꾸면 출력이 얼마나 바뀌나 (기울기)",
        "· ==그레이디언트==: 텐서에 대한 도함수 → 손실이 가장 빨리 ==커지는== 방향",
        "  · 그래서 반대로 간다: `W -= learning_rate * gradient`",
        "· ==학습률==: 너무 작으면 느림 / 너무 크면 튕겨 나감",
        "· ==모멘텀==: 굴러가는 공의 관성처럼 → 지역 최솟값에서 빠져나오게 도움",
      ],
      end: "*결국, 손실 언덕에서 가장 가파른 내리막으로 한 걸음씩",
    },
    {
      id: "2.4-backprop", sec: "2.4.4", title: "역전파",
      fig: { h: 232, alt: "정방향으로 손실을 계산하고 연쇄 법칙으로 거꾸로 그레이디언트를 구한다", f: k => [
        k.text(130, 36, "정방향 →", { size: 12, color: "muted" }),
        k.oval(50, 80, 34, 24, "x"), k.arrow(86, 80, 126, 80),
        k.box(130, 58, 80, 44, "× W"), k.arrow(212, 80, 252, 80),
        k.box(256, 58, 80, 44, "+ b"), k.arrow(338, 80, 378, 80, { label: "y_pred" }),
        k.box(382, 58, 100, 44, "손실 함수", { fill: "yellow" }),
        k.oval(560, 80, 44, 24, "y_true"), k.arrow(514, 80, 486, 80),
        k.arrow(432, 56, 432, 26), k.text(442, 22, "loss 값", { size: 12 }),
        k.line(432, 104, 432, 140, { color: "red", dash: true }), k.line(296, 104, 296, 140, { color: "red", dash: true }), k.line(170, 104, 170, 140, { color: "red", dash: true }),
        k.arrow(432, 140, 300, 140, { color: "red", label: "∂loss/∂b", ly: 16, lc: "red" }),
        k.arrow(296, 140, 174, 140, { color: "red", label: "× 연쇄 법칙", ly: 16, lc: "red" }),
        k.text(130, 160, "← 역방향", { size: 12, color: "red" }),
        k.red(60, 200, "∂loss/∂W 완성 → 옵티마이저에게", { size: 18 }),
        k.red(600, 114, "프레임워크가 자동으로!", { size: 16 }),
        k.box(590, 126, 160, 96, "TF: GradientTape\n\nTorch: .backward()\n\nJAX: jax.grad(f)", { mono: true, size: 11, fill: "blue" }),
      ] },
      pts: [
        "· ==연쇄 법칙==: `f(g(x))' = f'(g(x)) · g'(x)`",
        "· ==계산 그래프==: 연산을 노드로 그린 그래프",
        "· ==역전파==: 손실에서 출발해 그래프를 ==거꾸로== 따라가며 연쇄 법칙 적용 → 각 가중치의 그레이디언트",
        "· 요즘은 손으로 안 함 → JAX · 텐서플로 · 파이토치가 ==자동 미분== (3장)",
      ],
      end: "*결국, 역전파 = 연쇄 법칙을 뒤에서부터 적용한 것",
    },
  ],
});
