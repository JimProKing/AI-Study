BOOK.add({
  no: 3,
  title: "텐서플로, 파이토치, JAX 그리고 케라스에 대한 소개",
  short: "TF · 파이토치 · JAX · 케라스",
  one: "케라스(조립 키트)로 쓰고, 백엔드(원자재)는 골라 쓴다",
  cards: [
    {
      id: "3.2-frameworks", sec: "3.2", title: "프레임워크 사이의 관계",
      fig: { h: 232, alt: "고수준 케라스가 저수준 백엔드 텐서플로, 파이토치, JAX 위에서 돈다", f: k => [
        k.box(120, 14, 520, 60, "케라스 (고수준)", { fill: "yellow", bold: true, sub: "층 · 손실 · 옵티마이저 · 지표 · 훈련 루프" }),
        k.arrow(380, 76, 380, 100, { label: "KERAS_BACKEND로 선택", lx: 90, ly: 0, lc: "red" }),
        k.box(120, 104, 160, 60, "텐서플로", { fill: "mint", sub: "저수준" }),
        k.box(300, 104, 160, 60, "파이토치", { fill: "mint", sub: "저수준" }),
        k.box(480, 104, 160, 60, "JAX", { fill: "mint", sub: "저수준" }),
        k.ctext(380, 186, "공통: 텐서 · 변수 · 텐서 연산 · 역전파(자동 미분)", { size: 13 }),
        k.red(652, 40, "조립식\n건축 키트", { size: 17 }), k.red(652, 132, "원자재", { size: 17 }),
        k.red(120, 218, "+ 넘파이에서도 돌지만 그레이디언트가 없어 훈련은 X", { size: 16 }),
      ] },
      pts: [
        "· ==저수준== (텐서플로 · 파이토치 · JAX): 텐서, 변수, 텐서 연산, 역전파",
        "· ==고수준== (케라스): 층, 손실 함수, 옵티마이저, 지표, 훈련 루프",
        "· 케라스는 혼자 못 돎 → ==백엔드 엔진== 필요. 셋 중 아무거나 고르면 됨",
        "· `os.environ[\"KERAS_BACKEND\"] = \"jax\"` 처럼 케라스를 import 하기 전에 지정",
      ],
      end: "*결국, 케라스 코드 하나로 세 백엔드 어디서든 돌린다",
    },
    {
      id: "3.3-compare", sec: "3.3~3.5", title: "텐서플로 vs 파이토치 vs JAX",
      fig: { h: 248, alt: "세 저수준 프레임워크의 장점과 단점 비교표", f: k => {
        const rows = [
          ["", "장점", "단점"],
          ["텐서플로", "그래프 모드 · XLA로 빠름\n배포 생태계 최고 (모바일 · 웹)", "API가 방대 · 넘파이와 불일치\n허깅 페이스 지원 약함"],
          ["파이토치", "즉시 실행 → 디버깅 쉬움\n허깅 페이스 일급 지원", "가장 느림\nAPI 불일치 (axis vs dim)"],
          ["JAX", "대부분 가장 빠름 · TPU 최적\n넘파이와 같은 API", "디버깅 어려움\n저수준 루프가 장황"],
        ];
        const cols = [[20, 120], [140, 300], [440, 300]], hs = [34, 64, 64, 64];
        const out = []; let y = 14;
        rows.forEach((r, i) => { r.forEach((c, j) => out.push(k.box(cols[j][0], y, cols[j][1], hs[i], c, { r: 0, size: i === 0 || j === 0 ? 14 : 12, bold: i === 0 || j === 0, fill: i === 0 ? "yellow" : j === 0 ? (i === 2 ? "orange" : "mint") : "paper" }))); y += hs[i]; });
        return out;
      } },
      pts: [
        "· 텐서플로: 문자열 텐서 · ==래그드 텐서==까지 있는 가장 완전한 기능",
        "· 파이토치: 사용하고 싶은 거의 모든 모델을 파이토치 버전으로 구할 수 있음 → 요즘 선택의 큰 이유",
        "  · `torch.compile()` 로 빨라질 수 있지만 아직 함정이 많음",
        "· JAX: ==함수형 · 상태 비저장== 철학 → 빠르게 컴파일, 큰 모델 · 많은 장치로 확장 쉬움",
      ],
      end: "*결국, 정답은 없고 장단점이 다르다. 케라스면 갈아타기 쉽다",
    },
    {
      id: "3.3-autodiff", sec: "3.3~3.5", title: "그레이디언트 구하는 3가지 방법",
      fig: { h: 230, alt: "텐서플로 GradientTape, 파이토치 backward, JAX grad로 같은 그레이디언트를 구하는 코드", f: k => [
        k.box(20, 14, 230, 30, "텐서플로", { fill: "mint", bold: true }),
        k.box(265, 14, 230, 30, "파이토치", { fill: "orange", bold: true }),
        k.box(510, 14, 230, 30, "JAX", { fill: "yellow", bold: true }),
        k.box(20, 48, 230, 110, "", { fill: "blue" }), k.text(30, 102, "with tf.GradientTape()\n        as tape:\n    loss = f(W)\ng = tape.gradient(\n        loss, W)", { mono: true, size: 12 }),
        k.box(265, 48, 230, 110, "", { fill: "blue" }), k.text(275, 102, "W = torch.tensor(...,\n    requires_grad=True)\nloss = f(W)\nloss.backward()\ng = W.grad", { mono: true, size: 12 }),
        k.box(510, 48, 230, 110, "", { fill: "blue" }), k.text(520, 102, "grad_fn = jax.grad(f)\ng = grad_fn(W)", { mono: true, size: 12 }),
        k.red(30, 184, "테이프로 기록", { size: 17 }), k.red(275, 184, "텐서가 거꾸로 추적", { size: 17 }), k.red(520, 184, "함수를 미분 함수로", { size: 17 }),
        k.red(20, 220, "셋 다 같은 일: 손실에 대한 가중치의 그레이디언트", { size: 17 }),
      ] },
      pts: [
        "· 텐서플로: `GradientTape` 블록 안 연산을 기록 → `tape.gradient()`",
        "· 파이토치: `requires_grad=True` 텐서 → `loss.backward()` → 결과는 `.grad` 에 저장",
        "· JAX: 상태가 없음 → 손실 '함수'를 `jax.grad` 에 넣어 그레이디언트 '함수'를 얻음 (`jax.value_and_grad` 는 손실도 같이)",
        "· 3장에서 셋 다 같은 선형 분류기를 밑바닥부터 구현해 봄",
      ],
      end: "*결국, 내 필기의 torch 코드는 이 중 가운데 길",
    },
    {
      id: "3.6-layer-model", sec: "3.6.2", title: "층과 모델",
      fig: { h: 232, alt: "층은 가중치와 call 연산을 담은 상자이고, 입력을 받아 출력을 낸다", f: k => [
        k.box(200, 20, 360, 150, ""), k.text(216, 42, "Layer (층)", { bold: true }),
        k.box(220, 60, 150, 90, "가중치", { sub: "W, b (상태)", fill: "yellow" }),
        k.box(390, 60, 150, 90, "call()", { sub: "relu(x·W + b)", fill: "mint", mono: true }),
        k.arrow(372, 105, 388, 105),
        k.oval(90, 105, 64, 26, "입력"), k.arrow(156, 105, 218, 105),
        k.arrow(542, 105, 604, 105), k.oval(670, 105, 64, 26, "출력"),
        k.red(216, 202, "build() = 첫 입력 크기를 보고\n가중치를 만듦 (자동 크기 추론)", { size: 17 }),
      ] },
      pts: [
        "· ==층==: 상태(가중치) + 연산을 담은 객체. 딥러닝의 기본 단위",
        "· 데이터마다 맞는 층: 벡터 → `Dense` / 시퀀스 → `LSTM`, 트랜스포머 / 이미지 → `Conv2D`",
        "· ==모델== = 층의 그래프. 가장 흔한 형태는 층을 쌓은 `Sequential`",
        "· 모델 구조 = ==가설 공간== → 구조를 고르는 것 = 가정을 고르는 것",
      ],
      end: "*결국, 층은 레고 블록, 모델은 블록 조립품",
    },
    {
      id: "3.6-compile-fit", sec: "3.6.4", title: "compile → fit → evaluate → predict",
      fig: { h: 222, alt: "케라스 모델 사용 순서: compile, fit, evaluate, predict", f: k => [
        k.flow(30, [
          { t: "compile()", sub: "optimizer · loss\n· metrics", mono: true },
          { t: "fit()", sub: "x, y, epochs,\nbatch_size", mono: true, fill: "yellow" },
          { t: "evaluate()", sub: "테스트 성능", mono: true },
          { t: "predict()", sub: "새 입력 → 예측", mono: true, fill: "mint" },
        ], { gap: 36, h: 72 }),
        k.red(20, 140, "학습 과정 설정", { size: 17 }),
        k.red(209, 146, "validation_data로\n검증 손실도 같이 봄", { size: 17 }),
        k.red(398, 140, "마지막에 한 번", { size: 17 }),
        k.red(587, 140, "배치로 나눠 예측", { size: 17 }),
        k.box(20, 180, 720, 34, "model.fit(x, y, epochs=5, batch_size=128, validation_data=(x_val, y_val))", { mono: true, size: 12, fill: "blue" }),
      ] },
      pts: [
        "· ==에포크==: 훈련 데이터 전체를 한 바퀴",
        "· ==배치 크기==: 그레이디언트 한 번 계산에 쓰는 샘플 수",
        "· ==검증 데이터==: 훈련에 안 쓰는 데이터 → 일반화 확인용",
        "· ==손실 함수 고르기==가 제일 중요: 문제와 안 맞으면 엉뚱한 걸 최적화 (4장 끝 표 참고)",
        "· `history = model.fit(...)` → `history.history` 에 에포크별 손실 · 지표가 쌓임",
      ],
      end: "*결국, compile로 설정 → fit으로 훈련 → predict로 사용",
    },
  ],
});
