BOOK.add({
  no: 13,
  title: "실전 문제 해결을 위한 모범 사례",
  short: "실전 모범 사례",
  one: "튜닝 · 앙상블로 성능 끝까지, 혼합 정밀도 · 다중 GPU로 빠르게",
  cards: [
    {
      id: "13.1-tuning", sec: "13.1.1", title: "하이퍼파라미터 최적화",
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
      id: "13.1-ensemble", sec: "13.1.2", title: "모델 앙상블",
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
      id: "13.2-scale", sec: "13.2", title: "대규모 모델 훈련",
      fig: { h: 236, alt: "혼합 정밀도로 숫자를 가볍게 하고, 여러 GPU에 배치를 나눠 그레이디언트를 평균한다", f: k => {
        const out = [
          k.text(20, 22, "① 혼합 정밀도", { bold: true, size: 14 }),
          k.box(20, 40, 240, 26, "float32 (32비트)", { size: 12 }),
          k.box(20, 74, 120, 26, "float16", { size: 12, fill: "mint" }),
          k.red(150, 88, "절반! 빠름", { size: 16 }),
          k.text(20, 128, "mixed_precision.set_global_policy(\n    \"mixed_float16\")", { mono: true, size: 11 }),
          k.text(20, 172, "③ TPU", { bold: true, size: 14 }),
          k.text(20, 202, "TPUStrategy (코랩에서 사용)\nsteps_per_execution 으로 효율 ↑", { size: 12 }),
          k.text(310, 22, "② 다중 GPU (MirroredStrategy)", { bold: true, size: 14 }),
          k.box(310, 40, 420, 28, "배치 → 4등분", { fill: "yellow", size: 13 }),
          k.box(310, 160, 420, 28, "그레이디언트 평균 → 모든 복사본 동시 업데이트", { fill: "mint", size: 13 }),
          k.text(310, 216, "with strategy.scope(): model = ...", { mono: true, size: 11 }),
        ];
        for (let i = 0; i < 4; i++) {
          const x = 310 + i * 108;
          out.push(k.arrow(x + 48, 70, x + 48, 88), k.box(x, 90, 96, 46, "GPU " + i, { sub: "모델 복사본" }), k.arrow(x + 48, 138, x + 48, 158));
        }
        return out;
      } },
      pts: [
        "· ==혼합 정밀도==: 대부분 float16, 민감한 부분만 float32 → GPU에서 사실상 공짜 속도 향상",
        "· ==데이터 병렬화==: 모델 복사본마다 배치의 다른 조각 → 그레이디언트 평균",
        "· `MirroredStrategy`: 한 컴퓨터의 여러 GPU",
        "· ==TPU==: 구글 전용 하드웨어, 코랩에서 무료로 써 볼 수 있음",
      ],
      end: "*결국, 숫자는 가볍게, 일은 나눠서 빠르게",
    },
  ],
});
