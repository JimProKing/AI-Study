BOOK.add({
  no: 11,
  title: "텍스트를 위한 딥러닝",
  short: "텍스트",
  one: "단어를 숫자로 → 문맥을 보는 어텐션 → 트랜스포머",
  cards: [
    {
      id: "11.2-text-prep", sec: "11.2", title: "텍스트 데이터 준비",
      fig: { h: 206, alt: "원본 텍스트를 표준화, 토큰화, 인덱싱해서 숫자 벡터로 만든다", f: k => [
        k.flow(20, [
          { t: "원본 텍스트", shape: "oval" },
          { t: "① 표준화" },
          { t: "② 토큰화" },
          { t: "③ 인덱싱", fill: "yellow" },
          { t: "벡터", shape: "oval", fill: "mint" },
        ], { gap: 30, h: 52 }),
        ...[["\"I like Cats!\"", 80], ["\"i like cats\"", 230], ["[i] [like] [cats]", 380], ["[3, 27, 152]", 530], ["멀티-핫 / 임베딩", 680]].map(([t, x]) => k.ctext(x, 96, t, { mono: true, size: 12 })),
        k.red(20, 146, "TextVectorization 층 하나로 ①②③ 한 번에", { size: 18 }),
        k.text(20, 186, "어휘 사전: 0 = 패딩, 1 = OOV (모르는 단어)", { size: 13, color: "muted" }),
      ] },
      pts: [
        "· ==표준화==: 소문자로, 구두점 제거 → 의미 없는 차이 없애기",
        "· ==토큰화==: 단어 단위 / ==N-그램== / 문자 단위",
        "· ==인덱싱==: 토큰 → 정수. 자주 나오는 2~3만 개만 사용",
        "· `TextVectorization` 을 tf.data 파이프라인에 두면 GPU 학습과 병렬로 처리",
      ],
      end: "*결국, 글자 → 숫자로 바꿔야 모델이 먹는다",
    },
    {
      id: "11.3-bow-vs-seq", sec: "11.3", title: "BoW vs 시퀀스 모델",
      fig: { h: 228, alt: "단어 집합으로 보는 BoW와 순서대로 보는 시퀀스 모델 비교", f: k => [
        k.box(20, 14, 340, 40, "BoW (단어 집합)", { fill: "yellow", bold: true }),
        k.box(60, 70, 260, 110, "", { r: 34 }),
        k.text(90, 100, "the", { mono: true }), k.text(200, 94, "cat", { mono: true }), k.text(140, 140, "sat", { mono: true }),
        k.text(250, 150, "mat", { mono: true }), k.text(96, 160, "on", { mono: true }), k.text(240, 118, "the", { mono: true }),
        k.red(60, 206, "순서 무시, 빠름", { size: 17 }),
        k.box(400, 14, 340, 40, "시퀀스", { fill: "mint", bold: true }),
        k.cells(410, 90, ["the", "cat", "sat", "on", "the", "mat"], { cw: 54, size: 12, mono: true }),
        k.arrow(410, 140, 734, 140, { label: "순서대로", ly: 16 }),
        k.red(410, 206, "순서 유지, RNN · 트랜스포머", { size: 17 }),
      ] },
      pts: [
        "· BoW: ==멀티-핫== (유니그램 / 바이그램), ==TF-IDF== 가중치",
        "· 시퀀스: 정수 시퀀스 → ==임베딩== → RNN / 트랜스포머",
        "· 고르는 규칙: 훈련 샘플 수 ÷ 샘플당 평균 단어 수",
        "  · !!1,500보다 작으면 BoW, 크면 시퀀스!!",
      ],
      end: "*결국, 데이터 적고 글이 길면 BoW, 데이터 많고 글이 짧으면 시퀀스",
    },
    {
      id: "11.3-embedding", sec: "11.3.3", title: "단어 임베딩",
      fig: { h: 240, alt: "임베딩 공간에서 고양이→호랑이, 개→늑대가 같은 방향의 벡터로 표현된다", f: k => [
        k.plot(40, 20, 340, 190, { pts: [[0.2, 0.3, "고양이", "mint", 6], [0.2, 0.75, "호랑이", "mint", 6], [0.6, 0.3, "개", "orange", 6], [0.6, 0.75, "늑대", "orange", 6]] }),
        k.arrow(108, 147, 108, 76, { color: "red" }), k.arrow(244, 147, 244, 76, { color: "red" }),
        k.red(116, 112, "야생", { size: 16 }), k.red(252, 112, "야생", { size: 16 }),
        k.arrow(116, 153, 236, 153, { color: "teal", label: "고양이과 → 개과", lc: "teal", ly: 16 }),
        k.box(420, 24, 320, 80, "원-핫", { sub: "희소 · 고차원 (2만 차원)\n단어끼리 관계 없음" }),
        k.box(420, 120, 320, 80, "임베딩", { sub: "밀집 · 저차원 (256 차원)\n의미가 거리 · 방향으로", fill: "yellow" }),
        k.red(420, 224, "같은 방향 = 같은 의미 변화", { size: 17 }),
      ] },
      pts: [
        "· ==단어 임베딩==: 단어 사이 의미 관계를 벡터 사이의 거리 · 방향으로 표현",
        "· ① `Embedding` 층으로 작업과 함께 학습 ② 사전 훈련된 임베딩 (Word2Vec, GloVe) 불러오기",
        "· `mask_zero=True`: 패딩(0)을 무시하도록 마스킹",
      ],
      end: "*결국, 비슷한 단어는 가까이, 같은 관계는 같은 방향",
    },
    {
      id: "11.4-self-attention", sec: "11.4.1", title: "셀프 어텐션",
      fig: { h: 236, alt: "station이 문장의 다른 단어들을 얼마나 참고하는지 선 굵기로 보여준다", f: k => {
        const words = ["the", "train", "left", "the", "station", "on", "time"], w = [0.05, 0.9, 0.2, 0.05, 1, 0.1, 0.15];
        const out = [];
        words.forEach((t, i) => {
          const x = 20 + i * 102;
          out.push(k.raw(`<line class="ln s-red" x1="${x + 46}" y1="74" x2="474" y2="150" style="stroke-width:${0.6 + 6 * w[i]};stroke-opacity:${0.25 + 0.75 * w[i]}"/>`));
        });
        words.forEach((t, i) => out.push(k.box(20 + i * 102, 40, 92, 32, t, { mono: true, fill: i === 4 ? "yellow" : "paper" })));
        out.push(k.box(374, 150, 200, 40, "새 'station' 벡터", { fill: "mint" }));
        out.push(k.text(20, 20, "선 굵기 = 어텐션 점수", { size: 12, color: "muted" }));
        out.push(k.red(20, 130, "train과 관련 깊음\n→ '기차역'으로 이해", { size: 17 }));
        out.push(k.red(20, 222, "점수 = 쿼리 · 키 유사도 → softmax → 값(value)의 가중 합", { size: 17 }));
        return out;
      } },
      pts: [
        "· 같은 단어도 문맥에 따라 의미가 다름 → ==문맥 인식 표현==",
        "· ==쿼리 · 키 · 값==: 쿼리와 키의 유사도 → 가중치 → 값을 가중 합",
        "· ==멀티 헤드 어텐션==: 여러 '헤드'로 나눠서 서로 다른 관계를 동시에 학습",
        "· 어텐션은 순서를 모름 → ==위치 임베딩==으로 위치 정보를 더함",
      ],
      end: "*결국, 어텐션 = 단어마다 '누구를 얼마나 볼지' 정하는 것",
    },
    {
      id: "11.4-transformer-encoder", sec: "11.4.3", title: "트랜스포머 인코더",
      fig: { h: 262, alt: "입력, 멀티 헤드 어텐션, Add와 층 정규화, Dense, Add와 층 정규화로 쌓인 인코더 블록", f: k => [
        k.box(200, 214, 280, 34, "입력 (토큰 + 위치 임베딩)", { size: 13 }),
        k.box(200, 160, 280, 32, "멀티 헤드 어텐션", { fill: "yellow", bold: true }),
        k.box(200, 116, 280, 28, "Add & LayerNorm", { fill: "mint", mono: true, size: 13 }),
        k.box(200, 68, 280, 32, "Dense 투영 (2개)"),
        k.box(200, 24, 280, 28, "Add & LayerNorm", { fill: "mint", mono: true, size: 13 }),
        k.arrow(340, 212, 340, 194), k.arrow(340, 158, 340, 146), k.arrow(340, 114, 340, 102), k.arrow(340, 66, 340, 54),
        k.path("M200 231H170V130H198", { color: "red" }),
        k.path("M480 130H510V38H482", { color: "red" }),
        k.red(70, 176, "잔차\n연결", { size: 16 }), k.red(518, 110, "잔차 연결", { size: 16 }),
        k.red(530, 176, "모든 단어가 서로를 봄", { size: 17 }),
        k.red(530, 84, "단어별로 따로 변환", { size: 17 }),
      ] },
      pts: [
        "· ==트랜스포머 인코더== = 멀티 헤드 어텐션 + Dense + 잔차 연결 + ==층 정규화==",
        "· 텍스트 분류엔 인코더만 사용",
        "· 층 정규화(LayerNormalization): 배치 전체가 아니라 시퀀스 하나 안에서 정규화",
        "· 위치 임베딩이 없으면 사실상 BoW처럼 동작",
      ],
      end: "*결국, 어텐션 + 잔차 + 정규화 + Dense를 블록으로 쌓은 것",
    },
    {
      id: "11.5-seq2seq", sec: "11.5", title: "시퀀스-투-시퀀스 (기계 번역)",
      fig: { h: 216, alt: "인코더가 영어 문장을 읽고 디코더가 스페인어 문장을 한 토큰씩 만든다", f: k => [
        k.oval(70, 70, 56, 24, "영어 문장"), k.arrow(128, 70, 146, 70),
        k.box(150, 40, 200, 60, "인코더", { fill: "mint", sub: "원문 전체를 읽음" }),
        k.arrow(352, 70, 410, 70, { label: "문맥" }),
        k.box(414, 40, 200, 60, "디코더", { fill: "yellow", sub: "한 토큰씩 생성" }),
        k.arrow(616, 70, 646, 70), k.oval(696, 70, 46, 24, "스페인어"),
        k.cells(414, 140, ["[start]", "hola", "amigo", "[end]"], { cw: 50, size: 11, mono: true }),
        k.arrow(514, 138, 514, 102, { label: "지금까지 만든 토큰", lx: 64, ly: 0, lc: "red" }),
        k.red(20, 198, "훈련: 정답 앞부분을 넣고 다음 토큰 맞히기 / 추론: 만든 토큰을 다시 넣기", { size: 15 }),
      ] },
      pts: [
        "· ==시퀀스-투-시퀀스==: 입력 시퀀스 → 출력 시퀀스 (번역, 요약, 질의응답)",
        "· RNN 버전: 인코더의 최종 상태를 디코더의 초기 상태로",
        "· 트랜스포머 버전: 디코더가 ==인코더 출력에 어텐션== + ==코잘 마스킹== (미래 토큰은 못 보게)",
      ],
      end: "*결국, 읽는 쪽(인코더)과 쓰는 쪽(디코더)의 협업",
    },
  ],
});
