/* 자동 생성: scratchpad nb2lab.py 로 실행한 노트북을 옮긴 기록 */
BOOK.addLab({
 "ch": 8,
 "title": "8장 실습 · 합성곱 손계산부터 사전 훈련 모델까지",
 "file": "projects/ch08-convnets/ch08_convnets.ipynb",
 "run": "2026-11-21 · JAX 백엔드 · CPU",
 "cells": [
  {
   "t": "md",
   "html": "<h3>8장 · 컴퓨터 비전을 위한 딥러닝 (컨브넷)</h3><p>합성곱 · 풀링을 <b>손으로 한 번 계산해 보고</b>, 케라스 층이 똑같이 계산하는지 확인한 다음,<br>컨브넷을 훈련하고 <b>안을 열어 봐요</b>. CPU로 전체 약 5~8분.</p><div class=\"lab-tw\"><table><thead><tr><th>절</th><th>내용</th><th>그림 노트 카드 · 내 필기</th></tr></thead><tbody><tr><td>1</td><td>합성곱 손계산 = <code>Conv2D</code> · 패딩 · 스트라이드</td><td>8.1.1 합성곱 연산</td></tr><tr><td>2</td><td>최대 풀링 손계산 = <code>MaxPooling2D</code> · 풀링이 없으면?</td><td>8.1.2 최대 풀링</td></tr><tr><td>3</td><td>MNIST 컨브넷 훈련 · 필터와 특성 맵 들여다보기</td><td>8.1</td></tr><tr><td>4</td><td>강아지 vs 고양이 모델 구조 (shape 따라가기)</td><td>8.2 컨브넷 구조</td></tr><tr><td>5</td><td>데이터 증강: 변환 보기 · 적은 데이터에서 효과 실험</td><td>8.2.5 데이터 증강</td></tr><tr><td>6</td><td>사전 훈련 모델: 동결 · 미세 튜닝을 MNIST로 재현 + Xception 구조</td><td>8.3 사전 훈련된 모델 활용</td></tr></tbody></table></div><p>📦 책의 강아지 vs 고양이 데이터는 캐글 다운로드(약 800MB)가 필요해서, 실습은 <b>MNIST로 같은 원리를 재현</b>하고 강아지 · 고양이 모델은 구조만 살펴봐요.</p><p>🔍 = 내부 들여다보기, 💬 = 설명 포인트.</p>"
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
   "html": "<h4>1. 합성곱 연산 손계산 (§8.1.1)</h4><p>그림 노트 카드의 숫자 그대로: 5×5 입력 * 3×3 필터 → 3×3 특성 맵.<br>창(3×3)을 한 칸씩 밀면서 <b>겹치는 칸끼리 곱해서 모두 더해요</b>.</p>"
  },
  {
   "t": "code",
   "src": "x = np.array([1, 0, 1, 1, 0,\n              0, 1, 1, 0, 1,\n              1, 0, 1, 0, 1,\n              0, 1, 0, 1, 0,\n              1, 1, 0, 1, 1], dtype=\"float32\").reshape(5, 5)\nw = np.array([1, 0, 1,\n              0, 1, 0,\n              1, 0, 1], dtype=\"float32\").reshape(3, 3)\n\ndef conv2d_by_hand(x, w):\n    H, W = x.shape; k = w.shape[0]\n    out = np.zeros((H - k + 1, W - k + 1), dtype=\"float32\")\n    for i in range(out.shape[0]):\n        for j in range(out.shape[1]):\n            patch = x[i:i + k, j:j + k]          # 창\n            out[i, j] = (patch * w).sum()        # 곱해서 합\n    return out\n\nfm = conv2d_by_hand(x, w)\nprint(\"손계산 특성 맵:\\n\", fm)\nprint(\"왼쪽 위 칸 = 첫 3×3 창 * 필터 =\", x[:3, :3].ravel().tolist(), \"·\", w.ravel().tolist(), \"→\", fm[0, 0])",
   "out": "손계산 특성 맵:\n [[5. 2. 3.]\n [1. 4. 2.]\n [4. 2. 4.]]\n왼쪽 위 칸 = 첫 3×3 창 * 필터 = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0] · [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0] → 5.0",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<p>✏️ <b>공책 검증 포인트</b>: 그림 노트 8.1.1 카드의 특성 맵은 처음에 <code>4 2 4 / 2 4 2 / 3 2 4</code>로 잘못 그려져 있었어요.<br>이 노트북의 손계산 · <code>Conv2D</code> 결과(<code>5 2 3 / 1 4 2 / 4 2 4</code>)로 찾아내서 카드를 고쳤어요. 공책에 옮겨 둔 숫자가 있으면 같이 고치세요.</p>"
  },
  {
   "t": "md",
   "html": "<h5>🔍 케라스 <code>Conv2D</code>도 똑같이 계산하나?</h5><p><code>Conv2D</code>의 커널 shape은 <code>(높이, 너비, 입력 채널, 필터 수)</code> = <code>(3, 3, 1, 1)</code>. 손으로 만든 필터를 그대로 넣어 봐요.<br>(입력은 <code>(배치, 높이, 너비, 채널)</code> = <code>(1, 5, 5, 1)</code>)</p>"
  },
  {
   "t": "code",
   "src": "conv = layers.Conv2D(filters=1, kernel_size=3, use_bias=False)\nconv.build((None, 5, 5, 1))\nprint(\"커널 shape:\", tuple(conv.kernel.shape))\nconv.set_weights([w.reshape(3, 3, 1, 1)])\nout = keras.ops.convert_to_numpy(conv(x.reshape(1, 5, 5, 1)))[0, :, :, 0]\nprint(out)\nprint(\"손계산과 같은가?\", np.allclose(out, fm))",
   "out": "커널 shape: (3, 3, 1, 1)\n[[5. 2. 3.]\n [1. 4. 2.]\n [4. 2. 4.]]\n손계산과 같은가? True",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h5>패딩 · 스트라이드 · 필터 수 → 출력 shape</h5><div class=\"lab-tw\"><table><thead><tr><th>설정</th><th>출력 크기 공식 (입력 n, 창 k, 스트라이드 s)</th></tr></thead><tbody><tr><td><code>padding=\"valid\"</code> (기본)</td><td>⌊(n − k) / s⌋ + 1</td></tr><tr><td><code>padding=\"same\"</code></td><td>⌈n / s⌉ — 가장자리에 0을 둘러 크기 유지 (s=1일 때)</td></tr></tbody></table></div>"
  },
  {
   "t": "code",
   "src": "img = np.random.default_rng(0).random((1, 28, 28, 1)).astype(\"float32\")\nfor kw in [dict(), dict(padding=\"same\"), dict(strides=2), dict(padding=\"same\", strides=2)]:\n    y = layers.Conv2D(32, 3, **kw)(img)\n    print(f\"Conv2D(32, 3, {', '.join(f'{k}={v!r}' for k, v in kw.items()) or '기본'}) → {tuple(y.shape)}\")\nprint(\"→ 마지막 축 32 = 필터 개수 = 특성 맵의 '깊이'\")",
   "out": "Conv2D(32, 3, 기본) → (1, 26, 26, 32)\nConv2D(32, 3, padding='same') → (1, 28, 28, 32)\nConv2D(32, 3, strides=2) → (1, 13, 13, 32)\nConv2D(32, 3, padding='same', strides=2) → (1, 14, 14, 32)\n→ 마지막 축 32 = 필터 개수 = 특성 맵의 '깊이'",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<p>💬 <b>설명 포인트</b>: Dense와 Conv2D가 배우는 패턴은 어떻게 다를까?</p><blockquote><p>Dense는 입력 <b>전체</b>를 한꺼번에 보는 <b>전역 패턴</b>, Conv는 작은 창 안의 <b>지역 패턴</b>을 배워요.<br>같은 필터를 이미지 모든 위치에 재사용하니까 ① 한 곳에서 배운 패턴을 어디서든 알아보고(<b>이동 불변성</b>) ② 층을 쌓으면 작은 패턴 → 큰 패턴(<b>공간적 계층 구조</b>)을 배워요.</p></blockquote>"
  },
  {
   "t": "md",
   "html": "<h4>2. 최대 풀링 손계산 (§8.1.2)</h4><p>카드의 4×4 특성 맵을 2×2 창, 스트라이드 2로 → 칸마다 <b>최댓값</b>만 남겨요.</p>"
  },
  {
   "t": "code",
   "src": "fmap = np.array([[1, 3, 2, 4],\n                 [5, 6, 1, 2],\n                 [7, 2, 9, 1],\n                 [1, 8, 3, 4]], dtype=\"float32\")\nby_hand = fmap.reshape(2, 2, 2, 2).max(axis=(1, 3))     # (블록행, 칸행, 블록열, 칸열)에서 칸 축들을 max\nprint(\"손계산:\\n\", by_hand)\n\npool = layers.MaxPooling2D(pool_size=2)\nk_out = keras.ops.convert_to_numpy(pool(fmap.reshape(1, 4, 4, 1)))[0, :, :, 0]\nprint(\"MaxPooling2D:\\n\", k_out, \"\\n같은가?\", np.allclose(by_hand, k_out))\nprint(\"풀링 층의 가중치 개수:\", len(pool.weights), \"← 배우는 게 없는 고정 연산\")",
   "out": "손계산:\n [[6. 4.]\n [8. 9.]]\nMaxPooling2D:\n [[6. 4.]\n [8. 9.]] \n같은가? True\n풀링 층의 가중치 개수: 0 ← 배우는 게 없는 고정 연산",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h5>🔍 풀링이 없으면? (§8.1.2의 <code>model_no_max_pool</code>)</h5><p>풀링을 빼고 Conv만 쌓으면 ① 특성 맵이 거의 안 줄어서 Flatten 뒤 Dense 파라미터가 <b>폭발</b>하고 ② 마지막 층도 입력의 <b>작은 영역</b>만 봐요.</p>"
  },
  {
   "t": "code",
   "src": "def mnist_convnet(pool=True):\n    inputs = keras.Input(shape=(28, 28, 1))\n    x = layers.Conv2D(32, 3, activation=\"relu\")(inputs)\n    if pool: x = layers.MaxPooling2D(2)(x)\n    x = layers.Conv2D(64, 3, activation=\"relu\")(x)\n    if pool: x = layers.MaxPooling2D(2)(x)\n    x = layers.Conv2D(128, 3, activation=\"relu\")(x)\n    x = layers.Flatten()(x)\n    outputs = layers.Dense(10, activation=\"softmax\")(x)\n    return keras.Model(inputs, outputs)\n\ndef receptive_field(pool):\n    # 각 층을 지날 때 '마지막 층 한 칸이 보는 입력 영역' 크기\n    rf, jump = 1, 1\n    for kind in ([\"c\", \"p\", \"c\", \"p\", \"c\"] if pool else [\"c\", \"c\", \"c\"]):\n        if kind == \"c\": rf += 2 * jump\n        else: rf += jump; jump *= 2\n    return rf\n\nfor p in (True, False):\n    m = mnist_convnet(p)\n    last = [l for l in m.layers if isinstance(l, layers.Conv2D)][-1]\n    print(f\"풀링 {'있음' if p else '없음'}: 마지막 Conv 출력 {tuple(last.output.shape[1:])}, \"\n          f\"Flatten 길이 {m.layers[-2].output.shape[-1]:6d}, 전체 파라미터 {m.count_params():8,d}, \"\n          f\"마지막 층 한 칸이 보는 입력 {receptive_field(p)}×{receptive_field(p)}\")",
   "out": "풀링 있음: 마지막 Conv 출력 (3, 3, 128), Flatten 길이   1152, 전체 파라미터  104,202, 마지막 층 한 칸이 보는 입력 18×18\n풀링 없음: 마지막 Conv 출력 (22, 22, 128), Flatten 길이  61952, 전체 파라미터  712,202, 마지막 층 한 칸이 보는 입력 7×7",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>3. MNIST 컨브넷 훈련 (§8.1)</h4><p>2장의 Dense 모델(정확도 약 97.8%)과 비교해 봐요. 시간을 아끼려고 훈련 샘플 2만 개, 3 에포크만 써요.</p>"
  },
  {
   "t": "code",
   "src": "from keras.datasets import mnist\n(train_images, train_labels), (test_images, test_labels) = mnist.load_data()\ntrain_images = train_images.reshape((60000, 28, 28, 1)).astype(\"float32\") / 255\ntest_images = test_images.reshape((10000, 28, 28, 1)).astype(\"float32\") / 255\n\nmodel = mnist_convnet()\nmodel.compile(optimizer=\"rmsprop\", loss=\"sparse_categorical_crossentropy\", metrics=[\"accuracy\"])\nmodel.summary()\nmodel.fit(train_images[:20000], train_labels[:20000], epochs=3, batch_size=64, verbose=2)\ntest_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)\nprint(f\"테스트 정확도: {test_acc:.4f}\")",
   "out": "Model: \"functional_2\"\n\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃\n┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n│ input_layer_2 (InputLayer)      │ (None, 28, 28, 1)      │             0 │\n├─────────────────────────────────┼────────────────────────┼───────────────┤\n│ conv2d_11 (Conv2D)              │ (None, 26, 26, 32)     │           320 │\n├─────────────────────────────────┼────────────────────────┼───────────────┤\n│ max_pooling2d_3 (MaxPooling2D)  │ (None, 13, 13, 32)     │             0 │\n├─────────────────────────────────┼────────────────────────┼───────────────┤\n│ conv2d_12 (Conv2D)              │ (None, 11, 11, 64)     │        18,496 │\n├─────────────────────────────────┼────────────────────────┼───────────────┤\n│ max_pooling2d_4 (MaxPooling2D)  │ (None, 5, 5, 64)       │             0 │\n├─────────────────────────────────┼────────────────────────┼───────────────┤\n│ conv2d_13 (Conv2D)              │ (None, 3, 3, 128)      │        73,856 │\n├─────────────────────────────────┼────────────────────────┼───────────────┤\n│ flatten_2 (Flatten)             │ (None, 1152)           │             0 │\n├─────────────────────────────────┼────────────────────────┼───────────────┤\n│ dense_2 (Dense)                 │ (None, 10)             │        11,530 │\n└─────────────────────────────────┴────────────────────────┴───────────────┘\n\n Total params: 104,202 (407.04 KB)\n\n Trainable params: 104,202 (407.04 KB)\n\n Non-trainable params: 0 (0.00 B)\n\nEpoch 1/3\n313/313 - 14s - 46ms/step - accuracy: 0.8982 - loss: 0.3218\nEpoch 2/3\n313/313 - 12s - 39ms/step - accuracy: 0.9746 - loss: 0.0815\nEpoch 3/3\n313/313 - 13s - 40ms/step - accuracy: 0.9838 - loss: 0.0511\n테스트 정확도: 0.9844",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h5>🔍 내부 들여다보기 ① 첫 번째 Conv 층의 필터 32개</h5><p>첫 층 커널 shape = <code>(3, 3, 1, 32)</code>. 3×3 필터 32개를 그림으로 그려 봐요. 대부분 <b>방향이 있는 모서리 검출기</b>처럼 생겼어요.</p>"
  },
  {
   "t": "code",
   "src": "k1 = keras.ops.convert_to_numpy(model.layers[1].kernel)     # layers[0]은 Input\nprint(\"커널 shape:\", k1.shape)\nfig, axes = plt.subplots(2, 16, figsize=(12, 1.8))\nfor i, ax in enumerate(axes.ravel()):\n    ax.imshow(k1[:, :, 0, i], cmap=\"bwr\", vmin=-abs(k1).max(), vmax=abs(k1).max()); ax.axis(\"off\")\nplt.suptitle(\"첫 Conv 층 필터 (빨강 +, 파랑 −)\"); plt.show()",
   "out": "커널 shape: (3, 3, 1, 32)",
   "imgs": [
    "lab/ch08/fig01.png"
   ]
  },
  {
   "t": "md",
   "html": "<h5>🔍 내부 들여다보기 ② 숫자 하나가 층을 지나며 어떻게 바뀌나</h5><p>함수형 모델이라 중간 층 출력을 한 번에 꺼낼 수 있어요 (7장). 층마다 채널 8개씩만 보여 줘요.</p>"
  },
  {
   "t": "code",
   "src": "conv_pool_layers = [l for l in model.layers if isinstance(l, (layers.Conv2D, layers.MaxPooling2D))]\nactivation_model = keras.Model(model.inputs, [l.output for l in conv_pool_layers])\nacts = activation_model.predict(test_images[:1], verbose=0)\n\nfig, axes = plt.subplots(len(acts) + 1, 8, figsize=(10, 7.5))\naxes[0, 0].imshow(test_images[0, :, :, 0], cmap=\"gray_r\")\nfor ax in axes[0]: ax.axis(\"off\")\naxes[0, 0].set_title(f\"입력 (정답 {test_labels[0]})\", loc=\"left\", fontsize=9)\nfor r, (layer, a) in enumerate(zip(conv_pool_layers, acts), start=1):\n    for c in range(8):\n        axes[r, c].imshow(a[0, :, :, c], cmap=\"viridis\"); axes[r, c].axis(\"off\")\n    axes[r, 0].set_title(f\"{layer.name} {a.shape[1:]}\", loc=\"left\", fontsize=9)\nplt.tight_layout(); plt.show()\nprint(\"→ 아래로 갈수록 크기는 작아지고(26→1… ) 채널은 많아지며, 사람 눈에는 점점 추상적\")",
   "out": "→ 아래로 갈수록 크기는 작아지고(26→1… ) 채널은 많아지며, 사람 눈에는 점점 추상적",
   "imgs": [
    "lab/ch08/fig02.png"
   ]
  },
  {
   "t": "md",
   "html": "<h4>4. 강아지 vs 고양이 모델 구조 따라가기 (§8.2)</h4><p>책의 모델을 그대로 만들어 <b>shape만</b> 확인해요 (훈련 데이터는 캐글 필요).<br>그림 노트 8.2 카드의 숫자 <code>180×180×3 → 89×89×32 → 43×43×64 → 20×20×128 → 9×9×256 → 7×7×256</code>과 맞춰 보세요.</p>"
  },
  {
   "t": "code",
   "src": "inputs = keras.Input(shape=(180, 180, 3))\nx = layers.Rescaling(1.0 / 255)(inputs)\nx = layers.Conv2D(32, 3, activation=\"relu\")(x)\nx = layers.MaxPooling2D(2)(x)\nx = layers.Conv2D(64, 3, activation=\"relu\")(x)\nx = layers.MaxPooling2D(2)(x)\nx = layers.Conv2D(128, 3, activation=\"relu\")(x)\nx = layers.MaxPooling2D(2)(x)\nx = layers.Conv2D(256, 3, activation=\"relu\")(x)\nx = layers.MaxPooling2D(2)(x)\nx = layers.Conv2D(256, 3, activation=\"relu\")(x)\nx = layers.Flatten()(x)\noutputs = layers.Dense(1, activation=\"sigmoid\")(x)\ncats_dogs = keras.Model(inputs, outputs)\n\nfor l in cats_dogs.layers:\n    if isinstance(l, (layers.Conv2D, layers.MaxPooling2D, layers.Flatten)):\n        print(f\"{l.__class__.__name__:13s} → {tuple(l.output.shape[1:])}\")\nprint(\"\\n파라미터:\", f\"{cats_dogs.count_params():,}\")",
   "out": "Conv2D        → (178, 178, 32)\nMaxPooling2D  → (89, 89, 32)\nConv2D        → (87, 87, 64)\nMaxPooling2D  → (43, 43, 64)\nConv2D        → (41, 41, 128)\nMaxPooling2D  → (20, 20, 128)\nConv2D        → (18, 18, 256)\nMaxPooling2D  → (9, 9, 256)\nConv2D        → (7, 7, 256)\nFlatten       → (12544,)\n\n파라미터: 991,041",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<p>💬 <b>설명 포인트</b>: 180 → 178 → 89 처럼 줄어드는 이유는?</p><blockquote><p>3×3 Conv(<code>valid</code>)가 가장자리 1칸씩 잃어서 −2, 풀링이 절반. 이걸 반복하면 <b>크기 ↓ 깊이 ↑</b> — 컨브넷의 공통 패턴이에요.</p></blockquote>"
  },
  {
   "t": "code",
   "src": "# 실제로 캐글 데이터를 받아 두었다면 (책 8.2.2): 폴더 → 데이터셋\nDATA_DIR = \"cats_vs_dogs_small\"          # train/ validation/ test/ 아래 cat/ dog/ 폴더\nif os.path.isdir(DATA_DIR):\n    from keras.utils import image_dataset_from_directory\n    train_dataset = image_dataset_from_directory(os.path.join(DATA_DIR, \"train\"), image_size=(180, 180), batch_size=32)\nelse:\n    print(f\"'{DATA_DIR}' 폴더가 없어서 건너뜀 (책 8.2.2 절차로 만들면 이 셀이 데이터셋을 만들어요)\")",
   "out": "'cats_vs_dogs_small' 폴더가 없어서 건너뜀 (책 8.2.2 절차로 만들면 이 셀이 데이터셋을 만들어요)",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>5. 데이터 증강 (§8.2.5)</h4><h5>5-1. 변환이 어떻게 생겼나</h5><p>책과 같은 증강 층을 그림 노트 카드의 고양이 그림에 적용해 봐요.</p>"
  },
  {
   "t": "code",
   "src": "# 카드의 고양이를 이미지로 그리기 (180×180×3)\nfig = plt.figure(figsize=(1.8, 1.8), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.axis(\"off\")\nax.add_patch(plt.Circle((0, -0.1), 0.6, color=\"#f2a65a\"))\nax.add_patch(plt.Polygon([[-0.55, 0.15], [-0.45, 0.85], [-0.1, 0.45]], color=\"#f2a65a\"))\nax.add_patch(plt.Polygon([[0.55, 0.15], [0.45, 0.85], [0.1, 0.45]], color=\"#f2a65a\"))\nax.add_patch(plt.Circle((-0.2, 0.0), 0.08, color=\"k\")); ax.add_patch(plt.Circle((0.2, 0.0), 0.08, color=\"k\"))\nax.add_patch(plt.Circle((0.38, 0.45), 0.09, color=\"red\"))                # 오른쪽 귀의 빨간 점 → 뒤집기 확인용\nfig.canvas.draw(); cat = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(\"float32\"); plt.close(fig)\nprint(\"고양이 이미지:\", cat.shape)\n\ndata_augmentation = keras.Sequential([\n    layers.RandomFlip(\"horizontal\"),\n    layers.RandomRotation(0.1),\n    layers.RandomZoom(0.2),\n])\n\nfig, axes = plt.subplots(2, 5, figsize=(10, 4.2))\naxes[0, 0].imshow(cat.astype(\"uint8\")); axes[0, 0].set_title(\"원본\")\nfor ax in axes.ravel()[1:]:\n    ax.imshow(keras.ops.convert_to_numpy(data_augmentation(cat[None], training=True))[0].astype(\"uint8\"))\nfor ax in axes.ravel(): ax.axis(\"off\")\nplt.suptitle(\"training=True: 부를 때마다 다른 변환 (빨간 점 위치를 보세요)\"); plt.show()\n\nsame = np.allclose(keras.ops.convert_to_numpy(data_augmentation(cat[None], training=False))[0], cat)\nprint(\"training=False (predict/evaluate) 일 때 원본 그대로?\", same)",
   "out": "고양이 이미지: (180, 180, 3)\ntraining=False (predict/evaluate) 일 때 원본 그대로? True",
   "imgs": [
    "lab/ch08/fig03.png"
   ]
  },
  {
   "t": "md",
   "html": "<p>💬 <b>설명 포인트</b>: MNIST 숫자에도 <code>RandomFlip(\"horizontal\")</code>을 써도 될까?</p><blockquote><p>안 돼요! 좌우가 뒤집힌 '3'은 '3'이 아니고, 뒤집힌 '7'은 다른 글자처럼 보여요. 증강은 <b>실제로 있을 법한</b> 변환이어야 해요.<br>숫자에는 작은 회전 · 확대 · 이동만 써요 ↓</p></blockquote>"
  },
  {
   "t": "code",
   "src": "digit_aug = keras.Sequential([\n    layers.RandomRotation(0.05),\n    layers.RandomZoom(0.1),\n    layers.RandomTranslation(0.1, 0.1),\n])\nfig, axes = plt.subplots(1, 8, figsize=(9, 1.4))\nfor i, ax in enumerate(axes):\n    img = train_images[0] if i == 0 else keras.ops.convert_to_numpy(digit_aug(train_images[:1], training=True))[0]\n    ax.imshow(img[:, :, 0], cmap=\"gray_r\"); ax.axis(\"off\")\naxes[0].set_title(\"원본\", fontsize=9); plt.show()",
   "out": "",
   "imgs": [
    "lab/ch08/fig04.png"
   ]
  },
  {
   "t": "md",
   "html": "<h5>5-2. 실험: 데이터가 적을 때 증강의 효과</h5><p>책은 강아지 · 고양이 <b>2,000장</b>으로 시작해요. 같은 상황을 MNIST <b>1,000장</b>으로 만들어서, 증강 + 드롭아웃 있는 모델과 없는 모델을 비교해요.</p>"
  },
  {
   "t": "code",
   "src": "small_x, small_y = train_images[:1000], train_labels[:1000]\nval_x, val_y = train_images[50000:55000], train_labels[50000:55000]\n\ndef small_convnet(augment):\n    inputs = keras.Input(shape=(28, 28, 1))\n    x = digit_aug(inputs) if augment else inputs          # 증강은 모델 맨 앞에\n    x = layers.Conv2D(32, 3, activation=\"relu\")(x)\n    x = layers.MaxPooling2D(2)(x)\n    x = layers.Conv2D(64, 3, activation=\"relu\")(x)\n    x = layers.MaxPooling2D(2)(x)\n    x = layers.Flatten()(x)\n    if augment: x = layers.Dropout(0.5)(x)\n    outputs = layers.Dense(10, activation=\"softmax\")(x)\n    m = keras.Model(inputs, outputs)\n    m.compile(optimizer=\"rmsprop\", loss=\"sparse_categorical_crossentropy\", metrics=[\"accuracy\"])\n    return m\n\nh_plain = small_convnet(False).fit(small_x, small_y, epochs=30, batch_size=32, validation_data=(val_x, val_y), verbose=0)\nh_aug = small_convnet(True).fit(small_x, small_y, epochs=30, batch_size=32, validation_data=(val_x, val_y), verbose=0)\nplot_history({\"증강 없음\": h_plain, \"증강 + 드롭아웃\": h_aug}, key=\"val_loss\", title=\"1,000장으로 훈련: 검증 손실\")\nfor name, h in [(\"증강 없음\", h_plain), (\"증강 + 드롭아웃\", h_aug)]:\n    print(f\"{name:9s} 훈련 정확도 {h.history['accuracy'][-1]:.3f} | 최고 검증 정확도 {max(h.history['val_accuracy']):.3f} | \"\n          f\"최저 val_loss {min(h.history['val_loss']):.3f} (에포크 {int(np.argmin(h.history['val_loss'])) + 1})\")",
   "out": "증강 없음     훈련 정확도 1.000 | 최고 검증 정확도 0.944 | 최저 val_loss 0.216 (에포크 11)\n증강 + 드롭아웃 훈련 정확도 0.895 | 최고 검증 정확도 0.953 | 최저 val_loss 0.153 (에포크 29)",
   "imgs": [
    "lab/ch08/fig05.png"
   ]
  },
  {
   "t": "md",
   "html": "<h4>6. 사전 훈련된 모델 활용 (§8.3)</h4><h5>6-1. MNIST로 재현: '남이 잘 배운 눈'을 빌려 쓰기</h5><ul><li><b>사전 훈련</b>: 숫자 <b>0~4</b>만으로 컨브넷을 충분히 훈련 (= ImageNet 역할)</li><li><b>새 작업</b>: 숫자 <b>5~9</b> 구분. 레이블 있는 데이터는 <b>200장</b>뿐 (= 강아지 · 고양이 역할)</li></ul><p>세 방법을 비교해요.</p><div class=\"lab-tw\"><table><thead><tr><th>방법</th><th>합성곱 기반</th><th>새 분류기</th></tr></thead><tbody><tr><td>ⓐ 밑바닥부터</td><td>랜덤 초기화, 학습</td><td>학습</td></tr><tr><td>ⓑ 특성 추출</td><td>사전 훈련 가중치, <b>동결</b></td><td>학습</td></tr><tr><td>ⓒ 미세 튜닝</td><td>ⓑ 다음, <b>상위 층만 풀어서</b> 작은 학습률로</td><td>학습</td></tr></tbody></table></div>"
  },
  {
   "t": "code",
   "src": "def conv_base_fn():\n    inputs = keras.Input(shape=(28, 28, 1))\n    x = layers.Conv2D(32, 3, activation=\"relu\", name=\"conv1\")(inputs)\n    x = layers.MaxPooling2D(2)(x)\n    x = layers.Conv2D(64, 3, activation=\"relu\", name=\"conv2\")(x)\n    x = layers.MaxPooling2D(2)(x)\n    x = layers.Conv2D(128, 3, activation=\"relu\", name=\"conv3\")(x)\n    x = layers.Flatten()(x)\n    return keras.Model(inputs, x, name=\"conv_base\")\n\n# 사전 훈련: 0~4\nsrc = train_labels[:30000] < 5\nconv_base = conv_base_fn()\npre = keras.Sequential([conv_base, layers.Dense(5, activation=\"softmax\")])\npre.compile(optimizer=\"rmsprop\", loss=\"sparse_categorical_crossentropy\", metrics=[\"accuracy\"])\npre.fit(train_images[:30000][src], train_labels[:30000][src], epochs=3, batch_size=64, verbose=0)\nprint(\"사전 훈련(0~4) 테스트 정확도:\", round(pre.evaluate(test_images[test_labels < 5], test_labels[test_labels < 5], verbose=0)[1], 4))\npretrained_weights = conv_base.get_weights()\n\n# 새 작업: 5~9 (레이블 0~4로 바꿈), 200장\ntgt_tr = np.where(train_labels[30000:] >= 5)[0][:200] + 30000\ntgt_te = test_labels >= 5\nx_new, y_new = train_images[tgt_tr], train_labels[tgt_tr] - 5\nx_new_te, y_new_te = test_images[tgt_te], test_labels[tgt_te] - 5\nprint(\"새 작업 훈련 샘플:\", x_new.shape[0], \"| 테스트:\", x_new_te.shape[0])",
   "out": "사전 훈련(0~4) 테스트 정확도: 0.9961\n새 작업 훈련 샘플: 200 | 테스트: 4861",
   "imgs": []
  },
  {
   "t": "code",
   "src": "def new_task_model(pretrained, trainable):\n    base = conv_base_fn()\n    if pretrained: base.set_weights(pretrained_weights)\n    base.trainable = trainable\n    m = keras.Sequential([base, layers.Dropout(0.5), layers.Dense(5, activation=\"softmax\")])\n    m.compile(optimizer=\"rmsprop\", loss=\"sparse_categorical_crossentropy\", metrics=[\"accuracy\"])\n    return m, base\n\nm_scratch, _ = new_task_model(pretrained=False, trainable=True)\nm_scratch.fit(x_new, y_new, epochs=20, batch_size=32, verbose=0)\n\nm_frozen, base_frozen = new_task_model(pretrained=True, trainable=False)\nm_frozen.fit(x_new, y_new, epochs=20, batch_size=32, verbose=0)\nacc_frozen = m_frozen.evaluate(x_new_te, y_new_te, verbose=0)[1]\n\n# 미세 튜닝: 분류기를 먼저 훈련한 m_frozen에서 이어서, 상위 층(conv3)만 풀고 작은 학습률로\nbase_frozen.trainable = True\nfor layer in base_frozen.layers:\n    layer.trainable = (layer.name == \"conv3\")\nm_frozen.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),\n                 loss=\"sparse_categorical_crossentropy\", metrics=[\"accuracy\"])   # trainable 바꾸면 다시 compile!\nm_frozen.fit(x_new, y_new, epochs=10, batch_size=32, verbose=0)\n\nprint(f\"ⓐ 밑바닥부터  테스트 정확도: {m_scratch.evaluate(x_new_te, y_new_te, verbose=0)[1]:.4f}\")\nprint(f\"ⓑ 특성 추출   테스트 정확도: {acc_frozen:.4f}\")\nprint(f\"ⓒ + 미세 튜닝 테스트 정확도: {m_frozen.evaluate(x_new_te, y_new_te, verbose=0)[1]:.4f}\")",
   "out": "ⓐ 밑바닥부터  테스트 정확도: 0.9360\nⓑ 특성 추출   테스트 정확도: 0.9480\nⓒ + 미세 튜닝 테스트 정확도: 0.9486",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<p>💬 <b>설명 포인트</b>: 결과 읽기</p><blockquote><ul><li>ⓑ &gt; ⓐ: 0~4에서 배운 '모서리 · 곡선' 특성이 5~9에도 통해요. 레이블 200장으로는 그 특성을 처음부터 배우기 어려워요.</li><li>ⓒ ≈ ⓑ: 학습률이 아주 작고(1e-5) 데이터가 200장뿐이라 미세 튜닝으로 얻는 건 작아요. 책의 강아지 · 고양이(2,000장, Xception)에서는 97.5% → 98% 정도로 조금 오르는 정도예요.</li></ul></blockquote>"
  },
  {
   "t": "md",
   "html": "<h5>🔍 내부 들여다보기: <code>trainable = False</code>는 정확히 무엇을 바꾸나</h5><p>동결하면 가중치가 <code>trainable_weights</code>에서 빠져서, 옵티마이저가 <b>업데이트할 목록에 아예 없어요</b>.</p>"
  },
  {
   "t": "code",
   "src": "base = conv_base_fn(); base.set_weights(pretrained_weights)\nm = keras.Sequential([base, layers.Dense(5, activation=\"softmax\")])\nprint(\"동결 전  훈련되는 가중치 텐서 수:\", len(m.trainable_weights))\nbase.trainable = False\nprint(\"동결 후  훈련되는 가중치 텐서 수:\", len(m.trainable_weights), \"(새 Dense의 kernel, bias만)\")\n\nm.compile(optimizer=\"rmsprop\", loss=\"sparse_categorical_crossentropy\")\nbefore = keras.ops.convert_to_numpy(base.get_layer(\"conv1\").kernel).copy()\nm.fit(x_new, y_new, epochs=1, verbose=0)\nafter = keras.ops.convert_to_numpy(base.get_layer(\"conv1\").kernel)\nprint(\"훈련 후 conv1 커널이 그대로인가?\", np.allclose(before, after))",
   "out": "동결 전  훈련되는 가중치 텐서 수: 8\n동결 후  훈련되는 가중치 텐서 수: 2 (새 Dense의 kernel, bias만)\n훈련 후 conv1 커널이 그대로인가? True",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<p>💬 <b>설명 포인트</b>: 미세 튜닝 전에 왜 <b>분류기를 먼저</b> 훈련해야 할까?</p><blockquote><p>새 Dense는 랜덤이라 처음엔 오차가 커요. 이때 합성곱 층까지 풀어 두면 그 <b>큰 그레이디언트</b>가 사전 학습된 표현을 망가뜨려요.<br>분류기가 먼저 자리를 잡은 뒤, <b>작은 학습률</b>로 상위 층만 살짝 조정해요.</p></blockquote><h5>6-2. 책의 Xception 합성곱 기반 구조</h5><p>ImageNet 가중치(약 80MB)는 <code>USE_IMAGENET = True</code>일 때만 받아요. 구조 · 동결 실습은 가중치 없이도 돼요.</p>"
  },
  {
   "t": "code",
   "src": "USE_IMAGENET = False\nconv_base = keras.applications.Xception(weights=\"imagenet\" if USE_IMAGENET else None,\n                                        include_top=False, input_shape=(180, 180, 3))\nprint(\"층 개수:\", len(conv_base.layers), \"| 파라미터:\", f\"{conv_base.count_params():,}\")\nprint(\"출력 특성 맵:\", tuple(conv_base.output.shape[1:]), \"← 강아지 · 고양이 사진 한 장이 이 특성으로 바뀜\")\n\nconv_base.trainable = False\nprint(\"특성 추출 (전체 동결) 훈련 가중치 텐서:\", len(conv_base.trainable_weights))\n\nconv_base.trainable = True\nfor layer in conv_base.layers[:-4]:\n    layer.trainable = False\nprint(\"미세 튜닝 (마지막 4개 층만 풀기) 훈련 가중치 텐서:\", len(conv_base.trainable_weights),\n      \"→\", [l.name for l in conv_base.layers[-4:]])",
   "out": "층 개수: 132 | 파라미터: 20,861,480\n출력 특성 맵: (6, 6, 2048) ← 강아지 · 고양이 사진 한 장이 이 특성으로 바뀜\n특성 추출 (전체 동결) 훈련 가중치 텐서: 0\n미세 튜닝 (마지막 4개 층만 풀기) 훈련 가중치 텐서: 4 → ['block14_sepconv1_act', 'block14_sepconv2', 'block14_sepconv2_bn', 'block14_sepconv2_act']",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<hr><h4>정리 · 스스로 설명해 보기</h4><ol><li>5×5 입력 * 3×3 필터(valid)의 출력 크기는? <code>same</code>이면? 스트라이드 2면? → ______</li><li>Conv 층의 두 가지 성질 (이동 불변성 · 공간적 계층 구조)을 그림으로 → ______</li><li>풀링을 빼면 무슨 일이 생기나? (파라미터 수 · 보는 영역) → ______</li><li>증강이 predict 때 꺼지는 이유와, MNIST에 좌우 뒤집기를 안 쓰는 이유는? → ______</li><li>특성 추출 vs 미세 튜닝, 그리고 미세 튜닝의 순서는? → ______</li></ol>"
  }
 ]
});
