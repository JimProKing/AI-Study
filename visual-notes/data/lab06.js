/* 자동 생성: scratchpad nb2lab.py 로 실행한 노트북을 옮긴 기록 */
BOOK.addLab({
 "ch": 6,
 "title": "6장 실습 · '0 검출기'로 워크플로 끝까지 밟기",
 "file": "projects/ch06-workflow/ch06_workflow.ipynb",
 "run": "2026-11-21 · JAX 백엔드 · CPU",
 "cells": [
  {
   "t": "md",
   "html": "<h3>6장 · 일반적인 머신러닝 워크플로</h3><p>6장은 코드보다 <b>절차</b>가 중심인 장이에요. 그래서 작은 문제 하나를 정해 놓고<br><b>작업 정의 → 모델 개발 → 배포</b> 세 단계를 처음부터 끝까지 실제로 밟아 봐요.</p><blockquote><p>🎯 <b>오늘의 가상 프로젝트</b>: \"우편물 분류 공장에서, 손글씨 <b>0</b>(특수 우편번호)만 골라내는 검출기를 만들어 주세요.\"<br>→ MNIST로 <b>'0인가 아닌가'</b> 이진 분류. 양성이 약 10%뿐인 <b>불균형</b> 문제라서 6장의 함정이 다 드러나요.</p></blockquote><div class=\"lab-tw\"><table><thead><tr><th>단계</th><th>절</th><th>이 노트북에서 하는 일</th><th>그림 노트 카드 · 내 필기</th></tr></thead><tbody><tr><td>① 작업 정의</td><td>§6.1</td><td>입력 · 출력 정하기, 데이터 이해, <b>지표 고르기 (정확도의 함정, ROC · AUC)</b></td><td>6.1 작업 정의</td></tr><tr><td>② 모델 개발</td><td>§6.2</td><td>데이터 준비 → 평가 방법 → <b>기준점 넘기 → 과대적합 → 규제 · 튜닝</b></td><td>6.2 모델 개발 순서</td></tr><tr><td>③ 배포</td><td>§6.3</td><td>임계값 정하기 · 저장 · 다시 불러오기 · 속도 · <b>개념 이동 감시</b></td><td>6.3 모델 배포</td></tr></tbody></table></div><p>CPU로 전체 실행 약 3~5분. 🔍 = 내부 들여다보기, 💬 = 설명 포인트.</p>"
  },
  {
   "t": "md",
   "html": "<h4>0. 준비</h4><ul><li><code>KERAS_BACKEND</code>는 <b><code>import keras</code> 전에</b> 정해요. 바꾸려면 커널 재시작 → 이 셀부터 다시.</li><li>시드를 고정해서 실행할 때마다 거의 같은 결과가 나오게 해요 (CPU 연산 순서 때문에 소수점 끝자리는 조금 달라질 수 있어요).</li></ul>"
  },
  {
   "t": "code",
   "src": "import os\nos.environ[\"KERAS_BACKEND\"] = \"jax\"   # \"tensorflow\" / \"torch\" / \"jax\"\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport keras\nfrom keras import layers\n\nkeras.utils.set_random_seed(42)\nplt.rcParams[\"font.family\"] = \"Malgun Gothic\"     # 그래프 한글 깨짐 방지 (윈도)\nplt.rcParams[\"axes.unicode_minus\"] = False\nprint(\"keras\", keras.__version__, \"| backend =\", keras.backend.backend())",
   "out": "keras 3.15.1 | backend = jax",
   "imgs": []
  },
  {
   "t": "code",
   "src": "def plot_history(histories, key=\"val_loss\", title=None, ylim=None):\n    \"\"\"여러 history를 한 그래프에 겹쳐 그리기. histories = {\"이름\": history}\"\"\"\n    plt.figure(figsize=(7, 3.6))\n    for name, h in histories.items():\n        v = h.history[key]\n        plt.plot(range(1, len(v) + 1), v, \"-o\", ms=3, label=name)\n    plt.xlabel(\"에포크\"); plt.ylabel(key); plt.title(title or key)\n    if ylim: plt.ylim(*ylim)\n    plt.legend(); plt.grid(alpha=.3); plt.show()",
   "out": "",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h3>① 작업 정의 (§6.1)</h3><h4>1-1. 문제 정의</h4><div class=\"lab-tw\"><table><thead><tr><th>질문</th><th>이 프로젝트의 답</th></tr></thead><tbody><tr><td>입력은?</td><td>28×28 흑백 손글씨 이미지</td></tr><tr><td>예측할 것은?</td><td>이 숫자가 0인가? (예/아니오)</td></tr><tr><td>문제 유형은?</td><td><b>이진 분류</b> → 마지막 층 <code>Dense(1, sigmoid)</code> + <code>binary_crossentropy</code> (4장 정리표)</td></tr><tr><td>제약은?</td><td>놓치면(거짓 음성) 우편물이 잘못 배송됨 → <b>재현율이 중요</b></td></tr></tbody></table></div>"
  },
  {
   "t": "code",
   "src": "from keras.datasets import mnist\n\n(x_all, d_all), (x_test, d_test) = mnist.load_data()\ny_all = (d_all == 0).astype(\"float32\")       # 0이면 1(양성), 나머지 0\ny_test = (d_test == 0).astype(\"float32\")\nprint(\"이미지:\", x_all.shape, \"레이블 예:\", d_all[:10], \"→\", y_all[:10])",
   "out": "이미지: (60000, 28, 28) 레이블 예: [5 0 4 1 9 2 1 3 1 4] → [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>1-2. 데이터 이해 — 모델보다 먼저 데이터를 <b>본다</b></h4>"
  },
  {
   "t": "code",
   "src": "pos_rate = y_all.mean()\nprint(f\"양성(0) 비율 = {pos_rate:.3f}  → 10장 중 1장꼴. 불균형!\")\n\nfig, axes = plt.subplots(2, 8, figsize=(10, 2.8))\nfor ax, img in zip(axes[0], x_all[y_all == 1][:8]): ax.imshow(img, cmap=\"gray_r\"); ax.axis(\"off\")\nfor ax, img in zip(axes[1], x_all[y_all == 0][:8]): ax.imshow(img, cmap=\"gray_r\"); ax.axis(\"off\")\naxes[0, 0].set_title(\"양성 (0)\", loc=\"left\"); axes[1, 0].set_title(\"음성 (0 아님)\", loc=\"left\")\nplt.tight_layout(); plt.show()",
   "out": "양성(0) 비율 = 0.099  → 10장 중 1장꼴. 불균형!",
   "imgs": [
    "lab/ch06/fig01.png"
   ]
  },
  {
   "t": "code",
   "src": "# 평균 이미지: 클래스마다 '전형적인 모습'이 다른가?\nfig, axes = plt.subplots(1, 2, figsize=(5, 2.6))\naxes[0].imshow(x_all[y_all == 1].mean(axis=0), cmap=\"magma\"); axes[0].set_title(\"양성 평균\"); axes[0].axis(\"off\")\naxes[1].imshow(x_all[y_all == 0].mean(axis=0), cmap=\"magma\"); axes[1].set_title(\"음성 평균\"); axes[1].axis(\"off\")\nplt.show()\nprint(\"→ 양성 평균이 뚜렷한 고리 모양. 입력에 정답을 맞힐 정보가 '있다'는 신호\")",
   "out": "→ 양성 평균이 뚜렷한 고리 모양. 입력에 정답을 맞힐 정보가 '있다'는 신호",
   "imgs": [
    "lab/ch06/fig02.png"
   ]
  },
  {
   "t": "md",
   "html": "<h4>1-3. 성공 지표 고르기 — 정확도의 함정</h4><p>💬 <b>설명 포인트</b>: \"정확도 90% 모델\"은 좋은 모델일까?</p>"
  },
  {
   "t": "code",
   "src": "always_no = np.zeros_like(y_test)       # 무조건 '0 아님'이라고 답하는 '모델'\nacc = (always_no == y_test).mean()\nrecall = (always_no[y_test == 1] == 1).mean()\nprint(f\"무조건 '아니오' 정확도 = {acc:.3f}   ← 아무것도 안 했는데 90%!\")\nprint(f\"무조건 '아니오' 재현율 = {recall:.3f}   ← 0을 하나도 못 찾음\")",
   "out": "무조건 '아니오' 정확도 = 0.902   ← 아무것도 안 했는데 90%!\n무조건 '아니오' 재현율 = 0.000   ← 0을 하나도 못 찾음",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<blockquote><p>불균형 문제에서 정확도는 <b>거짓말을 해요</b>. 그래서 이렇게 봐요.</p><div class=\"lab-tw\"><table><thead><tr><th>지표</th><th>뜻</th><th>이 프로젝트에서</th></tr></thead><tbody><tr><td><b>정밀도</b></td><td>양성이라고 한 것 중 진짜 양성</td><td>잘못 골라낸 일반 우편물</td></tr><tr><td><b>재현율</b></td><td>진짜 양성 중 찾아낸 비율</td><td>놓친 특수 우편물 ← 더 비쌈</td></tr><tr><td><b>ROC AUC</b></td><td>임계값과 무관한 '구분 능력'</td><td>모델끼리 비교할 때</td></tr></tbody></table></div></blockquote>"
  },
  {
   "t": "md",
   "html": "<h3>② 모델 개발 (§6.2)</h3><h4>2-1. 데이터 준비 + 평가 방법 (홀드아웃)</h4><ul><li>섞어서 훈련 / 검증으로 나눠요. 테스트(<code>x_test</code>)는 <b>마지막에 딱 한 번</b>만 써요.</li><li>정규화(0~1)와 펴기(784)는 4장 · 5장과 같아요.</li></ul>"
  },
  {
   "t": "code",
   "src": "def prep(x):\n    return x.reshape((len(x), 784)).astype(\"float32\") / 255\n\nidx = np.random.default_rng(0).permutation(len(x_all))\nval_idx, tr_idx = idx[:10000], idx[10000:]\nx_tr, y_tr = prep(x_all[tr_idx]), y_all[tr_idx]\nx_val, y_val = prep(x_all[val_idx]), y_all[val_idx]\nx_te = prep(x_test)\nprint(\"훈련\", x_tr.shape, f\"양성 {y_tr.mean():.3f}\", \"| 검증\", x_val.shape, f\"양성 {y_val.mean():.3f}\")\nprint(\"→ 두 세트의 양성 비율이 비슷해야 검증이 '대표성'을 가짐\")",
   "out": "훈련 (50000, 784) 양성 0.099 | 검증 (10000, 784) 양성 0.100\n→ 두 세트의 양성 비율이 비슷해야 검증이 '대표성'을 가짐",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>2-2. 상식 수준의 기준점 넘기</h4><p>가장 단순한 모델(로지스틱 회귀 = <code>Dense(1, sigmoid)</code> 하나)부터 시작해요.<br>지표는 처음부터 <b>정밀도 · 재현율 · AUC</b>를 같이 달아요.</p>"
  },
  {
   "t": "code",
   "src": "def compile_(m):\n    # 지표 객체는 상태(누적 카운트)를 가지므로 모델마다 새로 만들어요\n    m.compile(optimizer=\"rmsprop\", loss=\"binary_crossentropy\",\n              metrics=[\"accuracy\", keras.metrics.Precision(name=\"precision\"),\n                       keras.metrics.Recall(name=\"recall\"), keras.metrics.AUC(name=\"auc\")])\n    return m\n\ndef show(h, tag):\n    last = {k: round(v[-1], 4) for k, v in h.history.items() if k.startswith(\"val_\")}\n    print(tag, last)\n\nbaseline = compile_(keras.Sequential([keras.Input(shape=(784,)), layers.Dense(1, activation=\"sigmoid\")]))\nh_base = baseline.fit(x_tr, y_tr, epochs=5, batch_size=256, validation_data=(x_val, y_val), verbose=0)\nshow(h_base, \"기준 모델\")",
   "out": "기준 모델 {'val_accuracy': 0.9904, 'val_auc': 0.995, 'val_loss': 0.032, 'val_precision': 0.9499, 'val_recall': 0.9538}",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>2-3. 과대적합할 수 있는 모델로 키우기</h4><p>층을 넓고 깊게, 에포크를 길게 → 검증 손실이 <b>바닥을 찍고 오르는</b> 지점을 찾아요. 그게 용량의 경계예요.</p>"
  },
  {
   "t": "code",
   "src": "big = compile_(keras.Sequential([keras.Input(shape=(784,)),\n                                 layers.Dense(512, activation=\"relu\"),\n                                 layers.Dense(512, activation=\"relu\"),\n                                 layers.Dense(1, activation=\"sigmoid\")]))\nh_big = big.fit(x_tr, y_tr, epochs=20, batch_size=256, validation_data=(x_val, y_val), verbose=0)\nshow(h_big, \"큰 모델\")\nplot_history({\"기준 모델\": h_base, \"큰 모델\": h_big}, key=\"val_loss\", title=\"큰 모델: 바닥을 찍고 다시 ↑ = 과대적합 도달\")\nbest = int(np.argmin(h_big.history[\"val_loss\"])) + 1\nprint(\"큰 모델 최저 val_loss 에포크:\", best)",
   "out": "큰 모델 {'val_accuracy': 0.9983, 'val_auc': 0.9999, 'val_loss': 0.0052, 'val_precision': 0.991, 'val_recall': 0.992}\n큰 모델 최저 val_loss 에포크: 13",
   "imgs": [
    "lab/ch06/fig03.png"
   ]
  },
  {
   "t": "md",
   "html": "<h4>2-4. 규제 + 하이퍼파라미터 튜닝</h4><p>작은 탐색(유닛 수 × 드롭아웃)을 돌려 <b>검증 AUC</b>로 고르고, 최저 손실 시점에서 멈추게 <code>EarlyStopping</code>을 써요.</p><p>⚠️ 탐색을 너무 많이 돌리면 <b>검증 세트에 과대적합</b>해요 (5장 정보 누설). 그래서 테스트는 여전히 손대지 않아요.</p>"
  },
  {
   "t": "code",
   "src": "results = []\nfor units in (64, 256):\n    for rate in (0.0, 0.3):\n        m = keras.Sequential([keras.Input(shape=(784,)), layers.Dense(units, activation=\"relu\"),\n                              layers.Dropout(rate), layers.Dense(units, activation=\"relu\"),\n                              layers.Dropout(rate), layers.Dense(1, activation=\"sigmoid\")])\n        compile_(m)\n        es = keras.callbacks.EarlyStopping(monitor=\"val_loss\", patience=3, restore_best_weights=True)\n        m.fit(x_tr, y_tr, epochs=20, batch_size=256, validation_data=(x_val, y_val), callbacks=[es], verbose=0)\n        r = m.evaluate(x_val, y_val, verbose=0, return_dict=True)\n        results.append((r[\"auc\"], r[\"recall\"], r[\"precision\"], units, rate, m))\n        print(f\"units={units:3d} dropout={rate:.1f} → val AUC {r['auc']:.4f} recall {r['recall']:.4f} precision {r['precision']:.4f}\")\n\nresults.sort(key=lambda t: -t[0])\n_, _, _, units, rate, final_model = results[0]\nprint(f\"\\n선택: units={units}, dropout={rate}\")",
   "out": "units= 64 dropout=0.0 → val AUC 0.9999 recall 0.9910 precision 0.9900\nunits= 64 dropout=0.3 → val AUC 0.9999 recall 0.9839 precision 0.9919\nunits=256 dropout=0.0 → val AUC 0.9999 recall 0.9910 precision 0.9860\nunits=256 dropout=0.3 → val AUC 0.9999 recall 0.9930 precision 0.9930\n\n선택: units=64, dropout=0.0",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>2-5. 🔍 ROC 곡선을 직접 그려 보기</h4><p><code>AUC</code> 지표가 숫자 하나로 요약한 걸, 임계값을 바꿔 가며 직접 계산해서 그려요.<br>x = <b>거짓 양성 비율</b>(FPR), y = <b>진짜 양성 비율</b>(TPR = 재현율).</p>"
  },
  {
   "t": "code",
   "src": "p_val = final_model.predict(x_val, verbose=0).ravel()\n\nthresholds = np.linspace(1, 0, 201)\ntpr = [((p_val >= t) & (y_val == 1)).sum() / (y_val == 1).sum() for t in thresholds]\nfpr = [((p_val >= t) & (y_val == 0)).sum() / (y_val == 0).sum() for t in thresholds]\nauc_manual = np.trapezoid(tpr, fpr)\n\nplt.figure(figsize=(4.2, 4))\nplt.plot(fpr, tpr, label=f\"최종 모델 (AUC ≈ {auc_manual:.4f})\")\nplt.plot([0, 1], [0, 1], \"--\", c=\"gray\", label=\"무작위 (0.5)\")\nfor t in (0.5, 0.1):\n    i = int(np.argmin(np.abs(thresholds - t)))\n    plt.scatter(fpr[i], tpr[i], zorder=3); plt.annotate(f\"임계값 {t}\", (fpr[i], tpr[i]), xytext=(8, -14), textcoords=\"offset points\")\nplt.xlabel(\"거짓 양성 비율 (FPR)\"); plt.ylabel(\"진짜 양성 비율 (재현율)\")\nplt.xlim(-0.01, 0.2); plt.title(\"왼쪽 위에 붙을수록 좋음 (x축 0~0.2 확대)\"); plt.legend(); plt.grid(alpha=.3); plt.show()",
   "out": "",
   "imgs": [
    "lab/ch06/fig04.png"
   ]
  },
  {
   "t": "md",
   "html": "<h4>2-6. 테스트는 마지막에 딱 한 번</h4>"
  },
  {
   "t": "code",
   "src": "r = final_model.evaluate(x_te, y_test, verbose=0, return_dict=True)\nprint({k: round(v, 4) for k, v in r.items()})",
   "out": "{'accuracy': 0.9975, 'auc': 0.9968, 'loss': 0.0107, 'precision': 0.9908, 'recall': 0.9837}",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h3>③ 배포 (§6.3)</h3><h4>3-1. 기대치 설정 — 숫자를 고객 언어로 번역</h4><p>\"정확도 99.x%\" 대신 <b>\"우편물 1만 통 중 0이 약 1천 통 있을 때, 몇 통 놓치고 몇 통 잘못 골라내는지\"</b>로 말해요.<br>그리고 <b>임계값</b>을 고객과 같이 정해요 (놓치는 비용 vs 잘못 고르는 비용).</p>"
  },
  {
   "t": "code",
   "src": "p_te = final_model.predict(x_te, verbose=0).ravel()\nprint(\"임계값 | 놓친 0 (FN) | 잘못 고른 우편물 (FP) | 재현율  | 정밀도\")\nfor t in (0.9, 0.5, 0.2, 0.05):\n    pred = p_te >= t\n    tp = (pred & (y_test == 1)).sum(); fn = (~pred & (y_test == 1)).sum(); fp = (pred & (y_test == 0)).sum()\n    print(f\"  {t:4.2f} | {fn:10d} | {fp:20d} | {tp/(tp+fn):.4f} | {tp/max(tp+fp,1):.4f}\")\nprint(\"→ 임계값을 낮출수록 덜 놓치지만 잘못 고르는 게 늘어남. 어디에 둘지는 '비용'이 정함\")",
   "out": "임계값 | 놓친 0 (FN) | 잘못 고른 우편물 (FP) | 재현율  | 정밀도\n  0.90 |         31 |                    3 | 0.9684 | 0.9968\n  0.50 |         16 |                    9 | 0.9837 | 0.9908\n  0.20 |         12 |                   17 | 0.9878 | 0.9827\n  0.05 |         10 |                   27 | 0.9898 | 0.9729\n→ 임계값을 낮출수록 덜 놓치지만 잘못 고르는 게 늘어남. 어디에 둘지는 '비용'이 정함",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>3-2. 모델 저장 → 다시 불러오기 (추론 서버가 하는 일)</h4><p><code>.keras</code> 파일 하나에 <b>구조 + 가중치 + 컴파일 설정</b>이 다 들어가요.</p>"
  },
  {
   "t": "code",
   "src": "import tempfile, time\npath = os.path.join(tempfile.gettempdir(), \"zero_detector.keras\")\nfinal_model.save(path)\nprint(\"파일 크기:\", round(os.path.getsize(path) / 1024), \"KB\")\n\nserved = keras.models.load_model(path)\nsame = np.allclose(served.predict(x_te[:100], verbose=0), final_model.predict(x_te[:100], verbose=0))\nprint(\"불러온 모델 예측이 원래와 같은가?\", same)",
   "out": "파일 크기: 453 KB\n불러온 모델 예측이 원래와 같은가? True",
   "imgs": []
  },
  {
   "t": "code",
   "src": "# 🔍 추론 속도: 한 장씩 vs 배치로 (서버 · 기기 배치를 고를 때 보는 숫자)\none = x_te[:1]\nserved.predict(one, verbose=0)                     # 워밍업 (첫 호출은 컴파일 때문에 느림)\nt0 = time.perf_counter()\nfor _ in range(50): served.predict(one, verbose=0)\nt_one = (time.perf_counter() - t0) / 50\nt0 = time.perf_counter(); served.predict(x_te, batch_size=1024, verbose=0); t_batch = (time.perf_counter() - t0) / len(x_te)\nprint(f\"한 장씩 predict(): {t_one*1000:.2f} ms/장   |   배치 predict(): {t_batch*1000:.4f} ms/장\")\nprint(\"→ 한 장씩은 호출 오버헤드가 큼. 서버는 요청을 모아 배치로 처리하기도 함\")\nprint(\"파라미터 수:\", served.count_params(), \"→ float32면 약\", round(served.count_params() * 4 / 1024), \"KB, int8 양자화면 약 1/4\")",
   "out": "한 장씩 predict(): 9.56 ms/장   |   배치 predict(): 0.0547 ms/장\n→ 한 장씩은 호출 오버헤드가 큼. 서버는 요청을 모아 배치로 처리하기도 함\n파라미터 수: 54465 → float32면 약 213 KB, int8 양자화면 약 1/4",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>3-3. 모니터링 · 유지 관리 — 개념 이동(concept drift)</h4><p>배포하고 몇 달 뒤, 공장에 <b>새 스캐너</b>가 들어와서 이미지가 조금 흐리고 어긋나게 들어온다고 해 봐요.<br>모델은 그대로인데 입력 분포가 바뀌면 성능이 <b>조용히</b> 떨어져요.</p>"
  },
  {
   "t": "code",
   "src": "def new_scanner(x, shift=3, noise=0.25, seed=0):\n    imgs = x.reshape(-1, 28, 28)\n    imgs = np.roll(imgs, shift, axis=2)                                 # 오른쪽으로 밀림\n    imgs = imgs + np.random.default_rng(seed).normal(0, noise, imgs.shape)  # 잡음\n    return np.clip(imgs, 0, 1).reshape(-1, 784).astype(\"float32\")\n\nx_drift = new_scanner(x_te)\nfig, axes = plt.subplots(1, 6, figsize=(8, 1.6))\nfor i, ax in enumerate(axes):\n    ax.imshow((x_te if i < 3 else x_drift)[i % 3].reshape(28, 28), cmap=\"gray_r\"); ax.axis(\"off\")\naxes[0].set_title(\"예전 스캐너\", loc=\"left\"); axes[3].set_title(\"새 스캐너\", loc=\"left\"); plt.show()\n\nfor name, x in [(\"예전 입력\", x_te), (\"새 스캐너 입력\", x_drift)]:\n    r = served.evaluate(x, y_test, verbose=0, return_dict=True)\n    print(f\"{name:8s} recall {r['recall']:.4f}  precision {r['precision']:.4f}  auc {r['auc']:.4f}\")",
   "out": "예전 입력    recall 0.9837  precision 0.9908  auc 0.9968\n새 스캐너 입력 recall 0.0000  precision 0.0000  auc 0.5115",
   "imgs": [
    "lab/ch06/fig05.png"
   ]
  },
  {
   "t": "md",
   "html": "<h5>🔍 레이블 없이도 이상을 눈치채는 법: 입력 · 출력 통계 감시</h5><p>배포 환경에선 정답 레이블이 바로 안 들어와요. 대신 <b>입력 픽셀 통계</b>와 <b>예측 양성 비율</b>을 매일 기록해 두면, 갑자기 변할 때 알람을 걸 수 있어요.</p>"
  },
  {
   "t": "code",
   "src": "for name, x in [(\"예전\", x_te), (\"새 스캐너\", x_drift)]:\n    p = served.predict(x, verbose=0).ravel()\n    print(f\"{name:5s} 픽셀 평균 {x.mean():.3f} | 픽셀 표준편차 {x.std():.3f} | 예측 양성 비율 {(p >= 0.5).mean():.3f}\")\nprint(\"→ 숫자가 달라졌다 = 새 데이터를 모아 레이블을 달고 재훈련할 때 (워크플로 ①로 돌아감)\")",
   "out": "예전    픽셀 평균 0.133 | 픽셀 표준편차 0.310 | 예측 양성 비율 0.097\n새 스캐너 픽셀 평균 0.206 | 픽셀 표준편차 0.295 | 예측 양성 비율 0.000\n→ 숫자가 달라졌다 = 새 데이터를 모아 레이블을 달고 재훈련할 때 (워크플로 ①로 돌아감)",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<hr><h4>정리 · 스스로 설명해 보기</h4><ol><li>이 프로젝트에서 정확도가 나쁜 지표인 이유는? → ______</li><li>ROC 곡선의 x축 · y축, 그리고 AUC 0.5의 뜻은? → ______</li><li>모델 개발 순서를 5단계로 (그림 노트 6.2 카드 보며) → ______</li><li>임계값은 누가 어떻게 정하나? → ______</li><li>개념 이동을 레이블 없이 눈치채는 방법은? → ______</li></ol>"
  }
 ]
});
