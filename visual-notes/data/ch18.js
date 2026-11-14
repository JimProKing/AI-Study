BOOK.add({
  no: 18,
  title: "실전 문제 해결을 위한 모범 사례",
  short: "실전 모범 사례",
  one: "튜닝 · 앙상블로 성능 끝까지, 다중 장치 · 낮은 정밀도로 빠르게",
  cards: [
    {
      id: "18.1-tuning", sec: "18.1.1", title: "하이퍼파라미터 최적화",
      fig: { h: 218, alt: "하이퍼파라미터를 고르고 모델을 만들어 훈련·검증하고 결과를 기록하는 반복 루프", f: k => [
        k.flow(30, [
          { t: "하이퍼파라미터 선택", fill: "yellow", size: 13 },
          { t: "모델 만들기" },
          { t: "훈련 · 검증" },
          { t: "결과 기록" },
        ], { gap: 36, h: 50 }),
        k.path("M663 82V110H96V82", { color: "teal" }),
        k.ctext(380, 124, "튜너가 다음 조합을 고름", { size: 12, color: "teal", bold: true }),
        k.box(20, 140, 470, 66, "def build_model(hp):\n    units = hp.Int(\"units\", min_value=16, max_value=64, step=16)\n    optimizer = hp.Choice(\"optimizer\", [\"rmsprop\", \"adam\"])", { mono: true, size: 11, fill: "blue" }),
        k.red(510, 166, "검증 세트에\n과대적합 주의!", { size: 18 }),
      ] },
      pts: [
        "· ==하이퍼파라미터==: 층 수, 유닛 수, 학습률, 드롭아웃 비율처럼 사람이 정하는 값",
        "· ==KerasTuner==: `RandomSearch`, `BayesianOptimization`, `Hyperband`",
        "· 좋은 조합을 찾으면 → 훈련 + 검증 데이터를 합쳐 다시 훈련",
      ],
      end: "*결국, 조합 찾기는 사람 대신 튜너에게",
    },
    {
      id: "18.1-ensemble", sec: "18.1.2", title: "모델 앙상블",
      fig: { h: 200, alt: "서로 다른 모델 셋의 예측을 가중 평균해 최종 예측을 낸다", f: k => [
        k.box(20, 20, 150, 44, "GBM (트리 계열)"), k.box(20, 80, 150, 44, "딥러닝 모델 1"), k.box(20, 140, 150, 44, "딥러닝 모델 2"),
        k.arrow(172, 42, 324, 90, { label: "× 0.5" }), k.arrow(172, 102, 320, 102, { label: "× 0.3" }), k.arrow(172, 162, 324, 114, { label: "× 0.2", ly: 14 }),
        k.circle(362, 102, 38, "가중 평균", { fill: "yellow", size: 13 }),
        k.arrow(402, 102, 470, 102), k.oval(560, 102, 80, 28, "최종 예측", { fill: "mint" }),
        k.red(470, 160, "핵심은 다양성!\n서로 다르게 틀려야 함", { size: 17 }),
      ] },
      pts: [
        "· ==앙상블==: 여러 모델의 예측을 합쳐 더 좋은 예측",
        "· 가중치는 검증 성능이 좋은 모델에 더 크게",
        "· 비슷한 모델 여러 개보다 ==서로 다른 종류==가 효과적 (예: 트리 계열 + 딥러닝)",
      ],
      end: "*결국, 여러 명이 코끼리를 만지면 전체 모습이 보인다",
    },
    {
      id: "18.2-multi-device", sec: "18.2", title: "다중 장치 훈련: 데이터 병렬 vs 모델 병렬",
      fig: { h: 240, alt: "데이터 병렬화는 모델을 복사하고 배치를 나누며, 모델 병렬화는 모델 자체를 장치에 나눈다", f: k => {
        const out = [k.text(20, 22, "① 데이터 병렬화", { bold: true, size: 14 }),
          k.box(20, 36, 340, 28, "배치 → 장치 수만큼 나눔", { fill: "yellow", size: 13 })];
        for (let i = 0; i < 3; i++) {
          const x = 20 + i * 116;
          out.push(k.arrow(x + 50, 66, x + 50, 82), k.box(x, 84, 104, 46, "GPU " + i, { sub: "모델 전체 복사" }), k.arrow(x + 50, 132, x + 50, 148));
        }
        out.push(k.box(20, 150, 340, 28, "그레이디언트 평균 → 동시 업데이트", { fill: "mint", size: 13 }));
        out.push(k.text(400, 22, "② 모델 병렬화", { bold: true, size: 14 }));
        out.push(k.box(400, 36, 340, 28, "모델이 너무 커서 한 장치에 안 들어감", { fill: "red", size: 13 }));
        ["층 1~4", "층 5~8", "층 9~12"].forEach((l, i) => out.push(k.box(400 + i * 116, 84, 104, 46, "GPU " + i, { sub: l })));
        out.push(k.arrow(506, 107, 514, 107), k.arrow(622, 107, 630, 107));
        out.push(k.red(400, 160, "모델의 변수 · 계산을 쪼갬", { size: 17 }));
        out.push(k.text(20, 214, "keras.distribution.set_distribution(keras.distribution.DataParallel([\"gpu:0\", \"gpu:1\"]))", { mono: true, size: 11 }));
        return out;
      } },
      pts: [
        "· ==데이터 병렬화==: 모델이 한 장치에 들어가면 → 복사본마다 배치의 다른 조각",
        "· ==모델 병렬화==: 큰 모델 → 변수와 계산을 여러 장치에 나눔",
        "· `keras.distribution` API로 백엔드에 상관없이 설정",
        "· ==TPU==: 구글 전용 하드웨어, 코랩에서도 써 볼 수 있음",
      ],
      end: "*결국, 데이터를 나누거나, 모델을 나누거나",
    },
    {
      id: "18.3-precision", sec: "18.3", title: "낮은 정밀도로 빠르게",
      fig: { h: 236, alt: "float32, float16, bfloat16, float8, int8 비트 수와 각각의 용도", f: k => {
        const rows = [["float32", 32, "기본 (안전)", "paper"], ["float16", 16, "혼합 정밀도 훈련 · float16 추론", "mint"], ["bfloat16", 16, "범위 넓음, TPU에 잘 맞음", "mint"], ["float8", 8, "더 나아간 훈련", "yellow"], ["int8", 8, "양자화 → 빠른 추론", "orange"]];
        const out = [];
        rows.forEach(([n, b, d, f], i) => {
          const y = 14 + i * 38;
          out.push(k.text(20, y + 15, n, { mono: true, size: 13, bold: true }));
          out.push(k.box(110, y, b * 8, 30, "", { fill: f, r: 0 }));
          out.push(k.text(110 + b * 8 + 10, y + 15, `${b}비트 · ${d}`, { size: 12 }));
        });
        out.push(k.text(20, 214, "keras.config.set_dtype_policy(\"mixed_float16\")", { mono: true, size: 12 }));
        out.push(k.red(520, 214, "비트 ↓ = 메모리 ↓ 속도 ↑", { size: 17 }));
        return out;
      } },
      pts: [
        "· ==혼합 정밀도==: 대부분 float16, 민감한 부분(가중치 등)은 float32 → GPU에서 사실상 공짜 속도 향상",
        "· ==손실 스케일링==: float16에서 작은 그레이디언트가 0이 되는 걸 막음",
        "· bfloat16: 정밀도는 낮지만 표현 범위가 넓음 → 둘 다 시도해 보기",
        "· ==양자화==: 가중치를 int8로 → 추론 속도 ↑",
      ],
      end: "*결국, 숫자를 가볍게 하면 빨라진다",
    },
  ],
});
