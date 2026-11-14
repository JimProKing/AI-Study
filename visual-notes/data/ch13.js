BOOK.add({
  no: 13,
  title: "시계열 예측",
  short: "시계열",
  one: "순서가 중요한 데이터 → 상태를 가진 순환 신경망",
  cards: [
    {
      id: "13.2-temperature", sec: "13.2", title: "온도 예측 문제",
      fig: { h: 228, alt: "지난 5일 데이터 창으로 24시간 뒤 온도를 예측한다", f: k => [
        k.plot(30, 24, 560, 160, { xl: "시간", yl: "온도",
          bands: [{ from: 0.18, to: 0.62, c: "yellow", label: "입력: 지난 5일 (시간당 1개 = 120 스텝)" }],
          curves: [{ f: t => 0.5 + 0.25 * Math.sin(t * 28) + 0.12 * Math.sin(t * 5), color: "ink", to: 0.62 }, { f: t => 0.5 + 0.25 * Math.sin(t * 28) + 0.12 * Math.sin(t * 5), color: "muted", dash: true, from: 0.62, to: 0.78 }],
          pts: [[0.76, 0.5 + 0.25 * Math.sin(0.76 * 28) + 0.12 * Math.sin(0.76 * 5), "24시간 뒤?", "red", 6, "red"]] }),
        k.box(610, 30, 130, 60, "기준점", { sub: "내일 = 오늘 같은 시각", fill: "mint", ss: 11 }),
        k.red(612, 124, "MAE 2.44°C\n이걸 넘어야 함", { size: 18 }),
      ] },
      pts: [
        "· 데이터: 예나 기후 (특성 14개, 10분 간격) → 시간당 1개로 샘플링",
        "· 분할: 훈련 50% / 검증 25% / 테스트 25%, ==시간 순서대로== (섞지 X)",
        "· `timeseries_dataset_from_array` 로 (입력 창, 타깃) 배치 생성",
        "· ==상식 수준의 기준점==: \"24시간 뒤 = 지금 온도\" → 검증 MAE 2.44°C",
        "· Dense, 1D Conv 모델은 기준점을 잘 못 넘음 → 순서 정보가 중요",
      ],
      end: "*결국, 기준점부터 정하고 그걸 넘는지로 판단",
    },
    {
      id: "13.3-rnn", sec: "13.3.1", title: "순환 신경망 (RNN)",
      fig: { h: 232, alt: "RNN 셀이 상태를 다음 타임스텝으로 넘기며 시퀀스를 처리한다", f: k => {
        const out = [k.box(30, 80, 100, 60, "RNN", { fill: "yellow", bold: true }), k.path("M130 96C170 96 170 50 80 50C40 50 50 70 60 78", { color: "red" }),
          k.red(140, 44, "상태", { size: 17 }), k.arrow(80, 182, 80, 142, { label: "x_t", lx: 18, ly: 0 }), k.arrow(80, 78, 80, 70),
          k.ctext(200, 112, "=", { size: 26, bold: true }), k.ctext(200, 140, "펼치면", { size: 12, color: "muted" })];
        ["t-1", "t", "t+1"].forEach((t, i) => {
          const x = 250 + i * 160;
          out.push(k.box(x, 80, 100, 60, "RNN", { fill: "yellow" }));
          out.push(k.arrow(x + 50, 182, x + 50, 142, { label: "x_" + t, lx: 22, ly: 0 }));
          out.push(k.arrow(x + 50, 78, x + 50, 40, { label: "출력_" + t, lx: 34, ly: 4 }));
          if (i < 2) out.push(k.arrow(x + 102, 110, x + 158, 110, { color: "red", label: "상태", lc: "red" }));
        });
        out.push(k.red(250, 218, "output_t = tanh(W·x_t + U·state_t + b)", { size: 17 }));
        return out;
      } },
      pts: [
        "· ==RNN==: 상태(state)를 가진 for 루프. 이전 스텝 정보를 다음 스텝으로 넘김",
        "· `SimpleRNN` 은 길어지면 ==그레이디언트 소실== → 실전 X",
        "· ==LSTM==: 정보를 실어 나르는 '이동 벨트'를 추가 → 먼 과거 정보 유지",
        "· ==GRU==: LSTM을 단순화, 더 가벼움",
        "· 입력 크기 `(batch, timesteps, features)`, `return_sequences=True` 면 매 스텝 출력",
      ],
      end: "*결국, RNN = 기억을 들고 다니며 순서대로 읽는 층",
    },
    {
      id: "13.3-advanced-rnn", sec: "13.3.4", title: "순환 신경망 고급 기법",
      fig: { h: 232, alt: "순환 드롭아웃, 스태킹 RNN, 양방향 RNN 세 가지", f: k => [
        k.text(20, 22, "① 순환 드롭아웃", { bold: true, size: 14 }),
        k.nn(40, 50, 160, 110, [3, 4, 3], { drop: ["1-1", "1-3"] }),
        k.text(20, 196, "LSTM(32, recurrent_dropout=0.25)", { mono: true, size: 11 }),
        k.text(270, 22, "② 스태킹", { bold: true, size: 14 }),
        k.box(280, 50, 160, 40, "LSTM", { fill: "mint" }), k.box(280, 110, 160, 40, "LSTM", { fill: "mint" }),
        k.arrow(360, 108, 360, 92), k.arrow(360, 172, 360, 152),
        k.text(270, 196, "return_sequences=True 필수", { mono: true, size: 11 }),
        k.text(500, 22, "③ 양방향", { bold: true, size: 14 }),
        k.cells(510, 92, ["x₁", "x₂", "x₃", "x₄"], { cw: 50 }),
        k.arrow(510, 76, 710, 76, { color: "teal", label: "정방향 RNN", lc: "teal" }),
        k.arrow(710, 146, 510, 146, { color: "red", label: "역방향 RNN", lc: "red", ly: 16 }),
        k.text(500, 196, "Bidirectional(LSTM(16))", { mono: true, size: 11 }),
      ] },
      pts: [
        "· ① 순환 층에 드롭아웃은 ==타임스텝마다 같은 마스크== → `recurrent_dropout`",
        "· ② ==스태킹==: 순환 층을 여러 개 쌓아 표현력 ↑ (계산 비용 ↑)",
        "· ③ ==양방향==: 앞→뒤, 뒤→앞 둘 다 읽고 합침. 텍스트엔 강력, 날씨처럼 최근이 중요한 시계열엔 별로",
        "· 드롭아웃 규제한 LSTM → 검증 MAE 2.27°C까지 개선 (기준점 2.44°C)",
      ],
      end: "*결국, 기준점 2.44 → 2.27. 순서를 존중하면 이긴다",
    },
  ],
});
