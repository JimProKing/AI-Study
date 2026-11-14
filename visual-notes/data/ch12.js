BOOK.add({
  no: 12,
  title: "객체 탐지",
  short: "객체 탐지",
  one: "물체마다 상자 + 라벨: 2단계(R-CNN) vs 1단계(YOLO · RetinaNet)",
  cards: [
    {
      id: "12.1-two-vs-one", sec: "12.1", title: "2단계 vs 1단계 탐지기",
      fig: { h: 214, alt: "R-CNN은 영역 제안 후 분류하는 2단계, YOLO와 RetinaNet은 한 번에 예측하는 1단계", f: k => [
        k.text(20, 40, "2단계", { bold: true, size: 15 }), k.text(20, 60, "R-CNN", { size: 12, color: "muted", mono: true }),
        k.flow(20, [
          { t: "이미지", shape: "oval", fill: "orange" },
          { t: "① 관심 영역 제안", sub: "상자 후보 많이" },
          { t: "② 영역마다 컨브넷", sub: "분류", fill: "mint" },
          { t: "상자 + 라벨", shape: "oval" },
        ], { x0: 110, gap: 28, h: 52 }),
        k.text(20, 140, "1단계", { bold: true, size: 15 }), k.text(20, 160, "YOLO · RetinaNet", { size: 12, color: "muted", mono: true }),
        k.flow(122, [
          { t: "이미지", shape: "oval", fill: "orange" },
          { t: "한 번에 상자 + 클래스 예측", fill: "yellow" },
          { t: "상자 + 라벨", shape: "oval" },
        ], { x0: 110, gap: 28, h: 52 }),
        k.red(110, 200, "1단계가 더 빠름 → 자율 주행 같은 실시간 용도", { size: 17 }),
      ] },
      pts: [
        "· ==객체 탐지==: 바운딩 박스로 물체를 찾고 라벨 붙이기 (약한 버전의 분할, 대신 훨씬 빠름)",
        "· ==2단계== R-CNN: 관심 영역을 먼저 제안 → 컨브넷으로 분류",
        "· ==1단계== YOLO, RetinaNet: 두 작업을 한 번에",
      ],
      end: "*결국, 정확도의 2단계, 속도의 1단계",
    },
    {
      id: "12.2-yolo", sec: "12.2", title: "YOLO 모델",
      fig: { h: 232, alt: "이미지를 그리드로 나누고 셀마다 상자와 신뢰도, 클래스 확률을 예측한다", f: k => [
        k.box(20, 20, 180, 180, "", { fill: "orange" }),
        k.grid(20, 20, 4, 4, null, { cw: 45, fill: "none", thin: true, hi: [5], hic: "yellow" }),
        k.box(52, 48, 90, 110, "", { fill: "none", color: "red", thick: true }),
        k.ctext(110, 218, "이미지를 그리드로", { size: 12, color: "muted" }),
        k.arrow(210, 110, 260, 70), k.arrow(210, 110, 260, 160),
        k.box(264, 40, 220, 60, "① 바운딩 박스 + 신뢰도", { fill: "yellow", sub: "셀마다 상자 하나 (책 버전)" }),
        k.box(264, 130, 220, 60, "② 클래스 확률 맵", { fill: "mint", sub: "이 영역은 어떤 물체?" }),
        k.text(510, 50, "IoU = 겹친 넓이 ÷ 합친 넓이", { size: 13, bold: true }),
        k.box(520, 70, 80, 60, "", { fill: "blue" }), k.box(560, 96, 80, 60, "", { fill: "red" }),
        k.red(510, 190, "신뢰도는 예측 상자와\n정답 상자의 IoU를 목표로", { size: 17 }),
      ] },
      pts: [
        "· 이미지를 ==그리드==로 나누고 각 셀에서 두 가지를 동시에 예측",
        "  · ① 바운딩 박스 + ==신뢰도 점수== ② 클래스 확률 맵",
        "· ==IoU==(Intersection over Union): 두 상자가 얼마나 겹치는지 0~1",
        "· 데이터: ==COCO== 데이터셋으로 밑바닥부터 훈련",
      ],
      end: "*결국, 한 번 보고(You Only Look Once) 상자와 이름을 같이 낸다",
    },
    {
      id: "12.3-retinanet", sec: "12.3", title: "RetinaNet과 특성 피라미드",
      fig: { h: 216, alt: "여러 층의 특성 맵을 합쳐 다양한 크기의 물체를 찾는 특성 피라미드 네트워크", f: k => {
        const out = [];
        [[40, 150], [90, 110], [130, 76], [162, 48]].forEach(([x, h], i) => out.push(k.box(x, 110 - h / 2, 32, h, "", { fill: "blue", r: 1 })));
        [[420, 48, "큰 물체"], [420, 90, "중간 물체"], [420, 132, "작은 물체"]].forEach(([x, y, l], i) => {
          out.push(k.box(x, y, 140, 34, l, { fill: i === 0 ? "orange" : i === 1 ? "yellow" : "mint", size: 13 }));
        });
        out.push(k.arrow(200, 70, 416, 62), k.arrow(200, 110, 416, 106), k.arrow(200, 150, 416, 148));
        out.push(k.ctext(110, 200, "컨브넷 여러 층", { size: 12, color: "muted" }));
        out.push(k.ctext(300, 196, "특성 결합 → 스케일별 특성 맵", { size: 12, color: "muted" }));
        out.push(k.red(580, 90, "크기가 다른\n물체도\n정확하게", { size: 17 }));
        return out;
      } },
      pts: [
        "· ==RetinaNet==: 1단계 탐지기 + ==특성 피라미드 네트워크==",
        "· 여러 합성곱 층의 특성을 합쳐 스케일이 다른 특성 맵을 만듦 → 크기가 다른 물체를 정확히",
        "· 사전 훈련 버전을 불러와 바로 사용 가능",
      ],
      end: "*결국, 여러 배율의 돋보기를 한꺼번에 쓰기",
    },
  ],
});
