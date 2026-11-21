/* 자동 생성: scratchpad nb2lab.py 로 실행한 노트북을 옮긴 기록 */
BOOK.addLab({
 "ch": 7,
 "title": "7장 실습 · 케라스 겉에서 속까지",
 "file": "projects/ch07-keras-deep-dive/ch07_keras_deep_dive.ipynb",
 "run": "2026-11-21 · 파이토치 백엔드 · CPU",
 "cells": [
  {
   "t": "md",
   "html": "<h3>7장 · 케라스 완전 정복</h3><p>케라스의 <b>겉(쉬운 API)부터 속(훈련 루프)까지</b> 한 겹씩 벗겨 보는 노트북이에요.<br>위에서부터 실행하면 '복잡성의 단계적 공개' 순서 그대로 내려가요. CPU로 전체 약 3~5분.</p><div class=\"lab-tw\"><table><thead><tr><th>절</th><th>내용</th><th>그림 노트 카드 · 내 필기</th></tr></thead><tbody><tr><td>1</td><td>모델을 만드는 3가지 방법 · 섞어 쓰기</td><td>7.1 복잡성의 단계적 공개</td></tr><tr><td>2</td><td>Sequential: 가중치는 <b>언제</b> 생기나 (<code>build</code>)</td><td>7.1</td></tr><tr><td>3</td><td>함수형 API: 고객 문의 티켓 (다중 입력 · 다중 출력) · 중간 층 꺼내기</td><td>7.2.2 함수형 API</td></tr><tr><td>4</td><td>서브클래싱 · 섞어 쓰기</td><td>7.1</td></tr><tr><td>5</td><td>내장 루프: 사용자 정의 지표 · 콜백 (EarlyStopping · Checkpoint · 직접 만든 콜백)</td><td>7.3.2 콜백</td></tr><tr><td>6</td><td>사용자 정의 훈련 루프 · 지표 저수준 사용 · <code>train_step()</code> 오버라이드</td><td>7.4 사용자 정의 훈련 루프</td></tr></tbody></table></div><h5>백엔드는 왜 파이토치?</h5><p>7.4의 훈련 루프는 <b>그레이디언트 계산 부분만</b> 백엔드마다 달라요. 이 노트북은 그 부분을 세 백엔드 모두 써 두었어요.<br>기본값은 루프가 가장 읽기 쉬운 <b>파이토치</b>(<code>loss.backward()</code>). 3장에서 JAX를 썼으니, 스터디 중에 <code>\"jax\"</code>나 <code>\"tensorflow\"</code>로 바꿔서 다시 돌려 보며 비교해 보세요.</p><p>🔍 = 내부 들여다보기, 💬 = 설명 포인트.</p>"
  },
  {
   "t": "md",
   "html": "<h4>0. 준비</h4><ul><li><code>KERAS_BACKEND</code>는 <b><code>import keras</code> 전에</b> 정해요. 바꾸려면 커널 재시작 → 이 셀부터 다시.</li><li>시드를 고정해서 실행할 때마다 거의 같은 결과가 나오게 해요 (CPU 연산 순서 때문에 소수점 끝자리는 조금 달라질 수 있어요).</li></ul>"
  },
  {
   "t": "code",
   "src": "import os\nos.environ[\"KERAS_BACKEND\"] = \"torch\"   # \"tensorflow\" / \"torch\" / \"jax\"\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport keras\nfrom keras import layers\n\nkeras.utils.set_random_seed(42)\nplt.rcParams[\"font.family\"] = \"Malgun Gothic\"     # 그래프 한글 깨짐 방지 (윈도)\nplt.rcParams[\"axes.unicode_minus\"] = False\nprint(\"keras\", keras.__version__, \"| backend =\", keras.backend.backend())",
   "out": "keras 3.15.1 | backend = torch",
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
   "html": "<h4>1. 복잡성의 단계적 공개 (§7.1)</h4><div class=\"lab-tw\"><table><thead><tr><th></th><th>쉬움 ←</th><th>→ 자유도 높음</th><th></th></tr></thead><tbody><tr><td><b>모델 만들기</b></td><td><code>Sequential</code></td><td>함수형 API</td><td>서브클래싱 (<code>keras.Model</code> 상속)</td></tr><tr><td><b>훈련하기</b></td><td><code>fit()</code></td><td><code>fit()</code> + 사용자 정의 지표 · 콜백 · <code>train_step()</code></td><td>완전히 직접 짠 루프</td></tr></tbody></table></div><ul><li>원칙: <b>쉬운 방법으로 시작</b>, 필요할 때만 한 단계씩 내려가요.</li><li>단계끼리 섞을 수 있어요. 예) 서브클래싱한 층을 함수형 모델 안에서 사용.</li></ul>"
  },
  {
   "t": "md",
   "html": "<h4>2. Sequential 모델 — 가중치는 언제 생기나 (§7.2.1)</h4><p>💬 <b>설명 포인트</b>: <code>Dense(64)</code>를 만들 때 가중치 행렬의 크기를 정할 수 있을까?</p><blockquote><p>못 정해요. 커널 shape = <code>(입력 특성 수, 64)</code>인데, 입력 특성 수를 아직 모르니까요.<br>그래서 케라스는 <b>처음 입력을 보는 순간</b> <code>build()</code>를 불러 가중치를 만들어요 (3장 '층과 모델' 카드).</p></blockquote>"
  },
  {
   "t": "code",
   "src": "model = keras.Sequential([\n    layers.Dense(64, activation=\"relu\"),\n    layers.Dense(10, activation=\"softmax\"),\n])\nprint(\"만든 직후 weights:\", model.weights)       # 아직 없음!\nprint(\"built?\", model.built)\n\nmodel.build(input_shape=(None, 3))               # 또는 실제 데이터를 한 번 넣기\nfor w in model.weights:\n    print(f\"{w.path:28s} {tuple(w.shape)}\")",
   "out": "만든 직후 weights: []\nbuilt? False\nsequential/dense/kernel      (3, 64)\nsequential/dense/bias        (64,)\nsequential/dense_1/kernel    (64, 10)\nsequential/dense_1/bias      (10,)",
   "imgs": []
  },
  {
   "t": "code",
   "src": "# 처음부터 입력 크기를 알려 주면 바로 만들어지고 summary()도 바로 볼 수 있어요\nmodel = keras.Sequential(name=\"my_example_model\")\nmodel.add(keras.Input(shape=(3,)))\nmodel.add(layers.Dense(64, activation=\"relu\", name=\"my_first_layer\"))\nmodel.add(layers.Dense(10, activation=\"softmax\", name=\"my_last_layer\"))\nmodel.summary()\nprint(\"파라미터 직접 계산:\", 3 * 64 + 64, \"+\", 64 * 10 + 10, \"=\", 3 * 64 + 64 + 64 * 10 + 10)",
   "out": "Model: \"my_example_model\"\n\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃\n┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n│ my_first_layer (Dense)          │ (None, 64)             │           256 │\n├─────────────────────────────────┼────────────────────────┼───────────────┤\n│ my_last_layer (Dense)           │ (None, 10)             │           650 │\n└─────────────────────────────────┴────────────────────────┴───────────────┘\n\n Total params: 906 (3.54 KB)\n\n Trainable params: 906 (3.54 KB)\n\n Non-trainable params: 0 (0.00 B)\n\n파라미터 직접 계산: 256 + 650 = 906",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>3. 함수형 API — 고객 문의 티켓 모델 (§7.2.2)</h4><p><b>입력 3개</b> → 합치기 → <b>출력 2개</b></p><div class=\"lab-tw\"><table><thead><tr><th>입력</th><th>shape</th><th>출력</th><th>활성화 · 손실</th></tr></thead><tbody><tr><td>제목 (멀티-핫)</td><td><code>(10000,)</code></td><td>우선순위 (0~1 점수)</td><td>sigmoid · <code>mean_squared_error</code></td></tr><tr><td>본문 (멀티-핫)</td><td><code>(10000,)</code></td><td>담당 부서 (4개 중 하나)</td><td>softmax · <code>categorical_crossentropy</code></td></tr><tr><td>태그 (멀티-핫)</td><td><code>(100,)</code></td><td></td><td></td></tr></tbody></table></div><p>층을 <b>함수처럼 호출</b>해서 선으로 잇는다고 생각하면 돼요: <code>x = layer(x)</code>.</p>"
  },
  {
   "t": "code",
   "src": "vocabulary_size, num_tags, num_departments = 10000, 100, 4\n\ntitle = keras.Input(shape=(vocabulary_size,), name=\"title\")\ntext_body = keras.Input(shape=(vocabulary_size,), name=\"text_body\")\ntags = keras.Input(shape=(num_tags,), name=\"tags\")\n\nfeatures = layers.Concatenate()([title, text_body, tags])\nfeatures = layers.Dense(64, activation=\"relu\", name=\"dense_features\")(features)\n\npriority = layers.Dense(1, activation=\"sigmoid\", name=\"priority\")(features)\ndepartment = layers.Dense(num_departments, activation=\"softmax\", name=\"department\")(features)\n\nmodel = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])\nmodel.summary()",
   "out": "Model: \"functional_3\"\n\n┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓\n┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃\n┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩\n│ title (InputLayer)  │ (None, 10000)     │          0 │ -                 │\n├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n│ text_body           │ (None, 10000)     │          0 │ -                 │\n│ (InputLayer)        │                   │            │                   │\n├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n│ tags (InputLayer)   │ (None, 100)       │          0 │ -                 │\n├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n│ concatenate         │ (None, 20100)     │          0 │ title[0][0],      │\n│ (Concatenate)       │                   │            │ text_body[0][0],  │\n│                     │                   │            │ tags[0][0]        │\n├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n│ dense_features      │ (None, 64)        │  1,286,464 │ concatenate[0][0] │\n│ (Dense)             │                   │            │                   │\n├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n│ priority (Dense)    │ (None, 1)         │         65 │ dense_features[0… │\n├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n│ department (Dense)  │ (None, 4)         │        260 │ dense_features[0… │\n└─────────────────────┴───────────────────┴────────────┴───────────────────┘\n\n Total params: 1,286,789 (4.91 MB)\n\n Trainable params: 1,286,789 (4.91 MB)\n\n Non-trainable params: 0 (0.00 B)",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h5>🔍 내부 들여다보기 ① <code>keras.Input</code>이 돌려주는 건 진짜 데이터가 아니다</h5><p><code>title</code>은 <b>심볼릭 텐서</b>(KerasTensor)예요. 값은 없고 shape · dtype만 있는 '자리 표시자'라서, 층을 호출하면 출력 shape만 계산해서 넘겨요.<br>그래서 데이터를 넣기도 전에 모델 그래프와 summary가 완성돼요.</p>"
  },
  {
   "t": "code",
   "src": "print(type(title).__name__, title.shape, title.dtype)\nprint(\"features:\", features.shape, \"| priority:\", priority.shape, \"| department:\", department.shape)\n\nfor layer in model.layers:\n    ins = layer.input if isinstance(layer.input, list) else [layer.input]\n    print(f\"{layer.name:16s} 입력 {[tuple(i.shape) for i in ins]} → 출력 {tuple(layer.output.shape)}\")",
   "out": "KerasTensor (None, 10000) float32\nfeatures: (None, 64) | priority: (None, 1) | department: (None, 4)\ntitle            입력 [] → 출력 (None, 10000)\ntext_body        입력 [] → 출력 (None, 10000)\ntags             입력 [] → 출력 (None, 100)\nconcatenate      입력 [(None, 10000), (None, 10000), (None, 100)] → 출력 (None, 20100)\ndense_features   입력 [(None, 20100)] → 출력 (None, 64)\npriority         입력 [(None, 64)] → 출력 (None, 1)\ndepartment       입력 [(None, 64)] → 출력 (None, 4)",
   "imgs": []
  },
  {
   "t": "code",
   "src": "# 가짜 데이터로 훈련 → 평가 → 예측 (입력 · 타깃을 '리스트'나 '이름 딕셔너리'로)\nnum_samples = 1280\nrng = np.random.default_rng(0)\ntitle_data = rng.integers(0, 2, size=(num_samples, vocabulary_size)).astype(\"float32\")\ntext_body_data = rng.integers(0, 2, size=(num_samples, vocabulary_size)).astype(\"float32\")\ntags_data = rng.integers(0, 2, size=(num_samples, num_tags)).astype(\"float32\")\npriority_data = rng.random(size=(num_samples, 1)).astype(\"float32\")\ndepartment_data = keras.utils.to_categorical(rng.integers(0, num_departments, size=num_samples), num_departments)\n\nmodel.compile(optimizer=\"adam\",\n              loss={\"priority\": \"mean_squared_error\", \"department\": \"categorical_crossentropy\"},\n              metrics={\"priority\": [\"mean_absolute_error\"], \"department\": [\"accuracy\"]})\nmodel.fit({\"title\": title_data, \"text_body\": text_body_data, \"tags\": tags_data},\n          {\"priority\": priority_data, \"department\": department_data}, epochs=1, verbose=2)\npriority_preds, department_preds = model.predict([title_data, text_body_data, tags_data], verbose=0)\nprint(\"priority_preds\", priority_preds.shape, \"department_preds\", department_preds.shape)",
   "out": "40/40 - 5s - 118ms/step - department_accuracy: 0.2625 - department_loss: 8.8321 - loss: 9.1529 - priority_loss: 0.3208 - priority_mean_absolute_error: 0.4903\npriority_preds (1280, 1) department_preds (1280, 4)",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<p>💬 <b>설명 포인트</b>: 출력이 2개면 손실은 어떻게 하나로 합칠까?</p><blockquote><p>각 출력의 손실을 따로 계산하고 <b>더해서</b>(기본 가중치 1) 하나의 스칼라로 만든 뒤 그 합을 최소화해요. <code>loss_weights</code>로 비중을 바꿀 수 있어요.</p></blockquote>"
  },
  {
   "t": "md",
   "html": "<h5>🔍 내부 들여다보기 ② 중간 층 출력 꺼내기 · 그래프 확장</h5><p>함수형 모델은 <b>층 그래프가 데이터 구조로 남아 있어서</b>, 중간 노드를 그대로 꺼내 새 모델을 만들 수 있어요.</p>"
  },
  {
   "t": "code",
   "src": "# 중간 특성(dense_features)을 출력으로 하는 '특성 추출기'\nfeature_extractor = keras.Model(inputs=model.inputs, outputs=model.get_layer(\"dense_features\").output)\nprint(\"특성 추출기 출력:\", feature_extractor.predict([title_data[:2], text_body_data[:2], tags_data[:2]], verbose=0).shape)\n\n# 기존 모델에 출력 하나 추가: 문제 난이도 (빠름 · 보통 · 어려움)\nf = model.layers[4].output\ndifficulty = layers.Dense(3, activation=\"softmax\", name=\"difficulty\")(f)\nnew_model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department, difficulty])\nprint(\"새 모델 출력 shape:\", [tuple(o.shape) for o in new_model.outputs])\n\n# 가중치를 공유하는가? (같은 층 객체를 재사용 → 같은 변수)\nprint(\"dense_features 커널을 공유?\", new_model.get_layer(\"dense_features\").kernel is model.get_layer(\"dense_features\").kernel)",
   "out": "특성 추출기 출력: (2, 64)\n새 모델 출력 shape: [(None, 1), (None, 4), (None, 3)]\ndense_features 커널을 공유? True",
   "imgs": []
  },
  {
   "t": "code",
   "src": "try:\n    keras.utils.plot_model(model, \"ticket_classifier.png\", show_shapes=True)\n    from IPython.display import Image, display; display(Image(\"ticket_classifier.png\"))\nexcept Exception as e:\n    print(\"plot_model 은 pydot + graphviz 설치가 필요해요 →\", type(e).__name__)\n    print(\"설치 후: pip install pydot  +  https://graphviz.org/download/\")",
   "out": "plot_model 은 pydot + graphviz 설치가 필요해요 → ImportError\n설치 후: pip install pydot  +  https://graphviz.org/download/",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>4. 서브클래싱 · 섞어 쓰기 (§7.2.3)</h4><ul><li><code>__init__()</code>에서 층을 만들고, <code>call()</code>에서 <b>파이썬 코드로 자유롭게</b> 정방향 계산을 써요.</li><li>대가: 층 그래프가 데이터 구조로 남지 않아요 → <b>호출 전엔 summary 불가</b>, 중간 층 꺼내기 불가.</li></ul>"
  },
  {
   "t": "code",
   "src": "class CustomerTicketModel(keras.Model):\n    def __init__(self, num_departments):\n        super().__init__()\n        self.concat_layer = layers.Concatenate()\n        self.mixing_layer = layers.Dense(64, activation=\"relu\")\n        self.priority_scorer = layers.Dense(1, activation=\"sigmoid\")\n        self.department_classifier = layers.Dense(num_departments, activation=\"softmax\")\n\n    def call(self, inputs):\n        features = self.concat_layer([inputs[\"title\"], inputs[\"text_body\"], inputs[\"tags\"]])\n        features = self.mixing_layer(features)\n        return self.priority_scorer(features), self.department_classifier(features)\n\nsub = CustomerTicketModel(num_departments=4)\nprint(\"호출 전 weights 개수:\", len(sub.weights))\np, d = sub({\"title\": title_data[:2], \"text_body\": text_body_data[:2], \"tags\": tags_data[:2]})\nprint(\"호출 후 weights 개수:\", len(sub.weights), \"| 출력\", tuple(p.shape), tuple(d.shape))",
   "out": "호출 전 weights 개수: 0\n호출 후 weights 개수: 6 | 출력 (2, 1) (2, 4)",
   "imgs": []
  },
  {
   "t": "code",
   "src": "# 섞어 쓰기: 서브클래싱한 층을 함수형 모델 안에서\nclass Classifier(keras.Model):\n    def __init__(self, num_classes=2):\n        super().__init__()\n        if num_classes == 2:\n            num_units, activation = 1, \"sigmoid\"\n        else:\n            num_units, activation = num_classes, \"softmax\"\n        self.dense = layers.Dense(num_units, activation=activation)\n\n    def call(self, inputs):\n        return self.dense(inputs)\n\ninputs = keras.Input(shape=(3,))\nx = layers.Dense(64, activation=\"relu\")(inputs)\noutputs = Classifier(num_classes=10)(x)\nmixed = keras.Model(inputs=inputs, outputs=outputs)\nprint(\"함수형 안의 서브클래싱 모델 → 출력\", mixed.output.shape)",
   "out": "함수형 안의 서브클래싱 모델 → 출력 (None, 10)",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>5. 내장 훈련 루프 확장하기 — 지표와 콜백 (§7.3)</h4><p>공통 실습: MNIST + 드롭아웃 있는 작은 모델 (책과 같은 <code>get_mnist_model</code>).</p>"
  },
  {
   "t": "code",
   "src": "from keras.datasets import mnist\n\ndef get_mnist_model():\n    inputs = keras.Input(shape=(28 * 28,))\n    features = layers.Dense(512, activation=\"relu\")(inputs)\n    features = layers.Dropout(0.5)(features)\n    outputs = layers.Dense(10, activation=\"softmax\")(features)\n    return keras.Model(inputs, outputs)\n\n(images, labels), (test_images, test_labels) = mnist.load_data()\nimages = images.reshape((60000, 28 * 28)).astype(\"float32\") / 255\ntest_images = test_images.reshape((10000, 28 * 28)).astype(\"float32\") / 255\ntrain_images, val_images = images[10000:], images[:10000]\ntrain_labels, val_labels = labels[10000:], labels[:10000]\nprint(train_images.shape, val_images.shape, test_images.shape)",
   "out": "(50000, 784) (10000, 784) (10000, 784)",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h5>5-1. 사용자 정의 지표 — 지표는 '상태를 가진 객체'</h5><p>지표는 배치마다 값을 <b>누적</b>했다가(<code>update_state</code>) 마지막에 계산해요(<code>result</code>). 그래서 변수(상태)를 가져요.</p>"
  },
  {
   "t": "code",
   "src": "from keras import ops\n\nclass RootMeanSquaredError(keras.metrics.Metric):\n    def __init__(self, name=\"rmse\", **kwargs):\n        super().__init__(name=name, **kwargs)\n        self.mse_sum = self.add_weight(name=\"mse_sum\", initializer=\"zeros\")\n        self.total_samples = self.add_weight(name=\"total_samples\", initializer=\"zeros\")\n\n    def update_state(self, y_true, y_pred, sample_weight=None):\n        y_true = ops.one_hot(y_true, num_classes=ops.shape(y_pred)[1])\n        mse = ops.sum(ops.square(y_true - y_pred))\n        self.mse_sum.assign_add(mse)\n        self.total_samples.assign_add(ops.shape(y_pred)[0])\n\n    def result(self):\n        return ops.sqrt(self.mse_sum / self.total_samples)\n\n    def reset_state(self):\n        self.mse_sum.assign(0.)\n        self.total_samples.assign(0.)\n\nmodel = get_mnist_model()\nmodel.compile(optimizer=\"rmsprop\", loss=\"sparse_categorical_crossentropy\",\n              metrics=[\"accuracy\", RootMeanSquaredError()])\nmodel.fit(train_images, train_labels, epochs=2, validation_data=(val_images, val_labels), verbose=2)\nprint(model.evaluate(test_images, test_labels, verbose=0, return_dict=True))",
   "out": "Epoch 1/2\n1563/1563 - 102s - 65ms/step - accuracy: 0.9129 - loss: 0.2946 - rmse: 0.3629 - val_accuracy: 0.9568 - val_loss: 0.1539 - val_rmse: 0.2585\nEpoch 2/2\n1563/1563 - 73s - 47ms/step - accuracy: 0.9546 - loss: 0.1597 - rmse: 0.2648 - val_accuracy: 0.9662 - val_loss: 0.1188 - val_rmse: 0.2279\n{'accuracy': 0.9682000279426575, 'loss': 0.10377020388841629, 'rmse': 0.21951118111610413}",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h5>5-2. 콜백 — 훈련을 지켜보다가 알아서 행동하는 감시자</h5><div class=\"lab-tw\"><table><thead><tr><th>콜백</th><th>하는 일</th></tr></thead><tbody><tr><td><code>EarlyStopping</code></td><td>지표가 <code>patience</code> 에포크 동안 안 좋아지면 <b>멈춤</b></td></tr><tr><td><code>ModelCheckpoint</code></td><td>에포크 끝마다 저장. <code>save_best_only=True</code>면 <b>최고일 때만</b></td></tr><tr><td>직접 만든 콜백</td><td><code>on_epoch_end</code>, <code>on_batch_end</code> 같은 시점에 원하는 코드 실행</td></tr></tbody></table></div>"
  },
  {
   "t": "code",
   "src": "import tempfile\nckpt_path = os.path.join(tempfile.gettempdir(), \"checkpoint_path.keras\")\ncallbacks_list = [\n    keras.callbacks.EarlyStopping(monitor=\"val_accuracy\", patience=1),\n    keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor=\"val_loss\", save_best_only=True, verbose=1),\n]\nmodel = get_mnist_model()\nmodel.compile(optimizer=\"rmsprop\", loss=\"sparse_categorical_crossentropy\", metrics=[\"accuracy\"])\nh = model.fit(train_images, train_labels, epochs=10, callbacks=callbacks_list,\n              validation_data=(val_images, val_labels), verbose=2)\nprint(f\"\\n10 에포크 중 {len(h.history['loss'])} 에포크에서 멈춤\")\nbest = keras.models.load_model(ckpt_path)\nprint(\"저장된 최고 모델 val_loss:\", round(best.evaluate(val_images, val_labels, verbose=0)[0], 4))",
   "out": "Epoch 1/10\n\nEpoch 1: val_loss improved from None to 0.15244, saving model to C:\\Users\\a\\AppData\\Local\\Temp\\checkpoint_path.keras\n\nEpoch 1: finished saving model to C:\\Users\\a\\AppData\\Local\\Temp\\checkpoint_path.keras\n1563/1563 - 33s - 21ms/step - accuracy: 0.9106 - loss: 0.2967 - val_accuracy: 0.9551 - val_loss: 0.1524\nEpoch 2/10\n\nEpoch 2: val_loss improved from 0.15244 to 0.11172, saving model to C:\\Users\\a\\AppData\\Local\\Temp\\checkpoint_path.keras\n\nEpoch 2: finished saving model to C:\\Users\\a\\AppData\\Local\\Temp\\checkpoint_path.keras\n1563/1563 - 21s - 14ms/step - accuracy: 0.9527 - loss: 0.1625 - val_accuracy: 0.9674 - val_loss: 0.1117\nEpoch 3/10\n\nEpoch 3: val_loss improved from 0.11172 to 0.10121, saving model to C:\\Users\\a\\AppData\\Local\\Temp\\checkpoint_path.keras\n\nEpoch 3: finished saving model to C:\\Users\\a\\AppData\\Local\\Temp\\checkpoint_path.keras\n1563/1563 - 13s - 8ms/step - accuracy: 0.9620 - loss: 0.1327 - val_accuracy: 0.9722 - val_loss: 0.1012\nEpoch 4/10\n\nEpoch 4: val_loss did not improve from 0.10121\n1563/1563 - 13s - 9ms/step - accuracy: 0.9691 - loss: 0.1114 - val_accuracy: 0.9740 - val_loss: 0.1014\nEpoch 5/10\n\nEpoch 5: val_loss improved from 0.10121 to 0.08926, saving model to C:\\Users\\a\\AppData\\Local\\Temp\\checkpoint_path.keras\n\nEpoch 5: finished saving model to C:\\Users\\a\\AppData\\Local\\Temp\\checkpoint_path.keras\n1563/1563 - 29s - 19ms/step - accuracy: 0.9712 - loss: 0.1018 - val_accuracy: 0.9771 - val_loss: 0.0893\nEpoch 6/10\n\nEpoch 6: val_loss did not improve from 0.08926\n1563/1563 - 46s - 29ms/step - accuracy: 0.9746 - loss: 0.0956 - val_accuracy: 0.9754 - val_loss: 0.0995\n\n10 에포크 중 6 에포크에서 멈춤\n저장된 최고 모델 val_loss: 0.0893",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h5>🔍 내부 들여다보기: 직접 만든 콜백으로 <b>배치마다</b> 손실 기록</h5><p><code>fit()</code>이 보여 주는 건 에포크 평균이에요. 콜백을 걸면 그 안의 배치 단위 움직임을 볼 수 있어요.</p>"
  },
  {
   "t": "code",
   "src": "class LossHistory(keras.callbacks.Callback):\n    def on_train_begin(self, logs=None):\n        self.per_batch_losses = []\n        self.epoch_ends = []\n\n    def on_batch_end(self, batch, logs=None):\n        self.per_batch_losses.append(logs.get(\"loss\"))   # 주의: 에포크 시작부터의 '누적 평균'\n\n    def on_epoch_end(self, epoch, logs=None):\n        self.epoch_ends.append(len(self.per_batch_losses))\n        print(f\"  [콜백] 에포크 {epoch + 1} 끝: 배치 {self.epoch_ends[-1]}개 지남, val_loss={logs['val_loss']:.4f}\")\n\nlh = LossHistory()\nmodel = get_mnist_model()\nmodel.compile(optimizer=\"rmsprop\", loss=\"sparse_categorical_crossentropy\", metrics=[\"accuracy\"])\nmodel.fit(train_images, train_labels, epochs=3, callbacks=[lh], validation_data=(val_images, val_labels), verbose=0)\n\nplt.figure(figsize=(7, 3.2))\nplt.plot(lh.per_batch_losses, lw=1)\nfor e in lh.epoch_ends[:-1]: plt.axvline(e, ls=\"--\", c=\"gray\")\nplt.xlabel(\"배치\"); plt.ylabel(\"loss (에포크 내 누적 평균)\"); plt.title(\"에포크가 바뀔 때마다 평균이 새로 시작된다\"); plt.show()",
   "out": "  [콜백] 에포크 1 끝: 배치 1563개 지남, val_loss=0.1644\n  [콜백] 에포크 2 끝: 배치 3126개 지남, val_loss=0.1314\n  [콜백] 에포크 3 끝: 배치 4689개 지남, val_loss=0.1074",
   "imgs": [
    "lab/ch07/fig01.png"
   ]
  },
  {
   "t": "md",
   "html": "<h4>6. 사용자 정의 훈련 루프 (§7.4)</h4><h5>6-1. <code>training=True / False</code></h5><p>드롭아웃 · 배치 정규화는 훈련과 추론에서 다르게 동작해요. 그래서 직접 짠 루프에선 <b>꼭</b> <code>model(x, training=True)</code>로 불러야 해요.</p>"
  },
  {
   "t": "code",
   "src": "m = get_mnist_model()\nx1 = val_images[:1]\na = ops.convert_to_numpy(m(x1, training=False)); b = ops.convert_to_numpy(m(x1, training=False))\nc = ops.convert_to_numpy(m(x1, training=True)); d = ops.convert_to_numpy(m(x1, training=True))\nprint(\"추론 모드 두 번 → 같은가?\", np.allclose(a, b))\nprint(\"훈련 모드 두 번 → 같은가?\", np.allclose(c, d), \"(드롭아웃 마스크가 매번 달라서)\")",
   "out": "추론 모드 두 번 → 같은가? True\n훈련 모드 두 번 → 같은가? False (드롭아웃 마스크가 매번 달라서)",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h5>6-2. 지표의 저수준 사용: <code>update_state()</code> → <code>result()</code> → <code>reset_state()</code></h5><p>5-1에서 만든 지표가 <code>fit()</code> 안에서 실제로 이렇게 쓰여요.</p>"
  },
  {
   "t": "code",
   "src": "metric = keras.metrics.SparseCategoricalAccuracy()\ntargets = np.array([0, 1, 2]); predictions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=\"float32\")\nmetric.update_state(targets, predictions)\nprint(\"1번 누적 후:\", float(metric.result()))\nmetric.update_state(np.array([0, 0, 0]), predictions)       # 이번엔 1/3만 맞음\nprint(\"2번 누적 후:\", float(metric.result()), \"← (3 + 1) / 6\")\nprint(\"내부 상태 변수:\", [(v.name, float(ops.convert_to_numpy(v))) for v in metric.variables])\nmetric.reset_state(); print(\"reset 후:\", float(metric.result()))\n\nmean_tracker = keras.metrics.Mean()\nfor v in [0, 1, 2, 3, 4]: mean_tracker.update_state(v)\nprint(\"Mean 지표:\", float(mean_tracker.result()))",
   "out": "1번 누적 후: 1.0\n2번 누적 후: 0.6666666865348816 ← (3 + 1) / 6\n내부 상태 변수: [('total', 4.0), ('count', 6.0)]\nreset 후: 0.0\nMean 지표: 2.0",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h5>6-3. 완전한 훈련 루프 직접 짜기 — <code>fit()</code>의 속</h5><p>그림 노트 7.4 카드의 5단계:<br><b>① 배치 뽑기 → ② 정방향 <code>model(x, training=True)</code> → ③ 손실 → ④ 그레이디언트 → ⑤ <code>optimizer.apply()</code></b></p><p>④만 백엔드마다 달라요. 아래 셀은 지금 백엔드(<code>keras.backend.backend()</code>)에 맞는 가지를 골라 실행해요.</p><div class=\"lab-tw\"><table><thead><tr><th>백엔드</th><th>④ 그레이디언트</th></tr></thead><tbody><tr><td>텐서플로</td><td><code>with tf.GradientTape() as tape:</code> → <code>tape.gradient(loss, weights)</code></td></tr><tr><td>파이토치</td><td><code>model.zero_grad()</code> → <code>loss.backward()</code> → <code>v.value.grad</code></td></tr><tr><td>JAX</td><td>상태를 <b>인자로 넘기는</b> <code>model.stateless_call()</code> + <code>jax.value_and_grad</code> (3장의 함수형 철학)</td></tr></tbody></table></div>"
  },
  {
   "t": "code",
   "src": "model = get_mnist_model()\nloss_fn = keras.losses.SparseCategoricalCrossentropy()\noptimizer = keras.optimizers.RMSprop()\nmetrics = [keras.metrics.SparseCategoricalAccuracy()]\nloss_tracking_metric = keras.metrics.Mean()\nBACKEND = keras.backend.backend()\n\nif BACKEND == \"torch\":\n    import torch\n\n    def train_step(inputs, targets):\n        predictions = model(inputs, training=True)              # ② 정방향\n        loss = loss_fn(targets, predictions)                    # ③ 손실\n        model.zero_grad()                                       # .grad 누적 방지 (3장!)\n        loss.backward()                                         # ④ 역전파\n        gradients = [v.value.grad for v in model.trainable_weights]\n        with torch.no_grad():\n            optimizer.apply(gradients, model.trainable_weights) # ⑤ 업데이트\n        return loss, predictions\n\nelif BACKEND == \"tensorflow\":\n    import tensorflow as tf\n\n    def train_step(inputs, targets):\n        with tf.GradientTape() as tape:                         # ④ 기록 시작\n            predictions = model(inputs, training=True)          # ②\n            loss = loss_fn(targets, predictions)                # ③\n        gradients = tape.gradient(loss, model.trainable_weights)\n        optimizer.apply(gradients, model.trainable_weights)     # ⑤\n        return loss, predictions\n\nelif BACKEND == \"jax\":\n    import jax\n\n    def compute_loss_and_updates(trainable_vars, non_trainable_vars, inputs, targets):\n        predictions, non_trainable_vars = model.stateless_call(\n            trainable_vars, non_trainable_vars, inputs, training=True)   # ② (상태를 인자로)\n        return loss_fn(targets, predictions), (predictions, non_trainable_vars)   # ③\n\n    grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)  # ④ 함수 → 그레이디언트 함수\n    optimizer.build(model.trainable_variables)\n    jax_state = [[v.value for v in model.trainable_variables],\n                 [v.value for v in model.non_trainable_variables],\n                 [v.value for v in optimizer.variables]]\n\n    def train_step(inputs, targets):\n        tv, ntv, ov = jax_state\n        (loss, (predictions, ntv)), grads = grad_fn(tv, ntv, inputs, targets)\n        tv, ov = optimizer.stateless_apply(ov, grads, tv)                  # ⑤ 새 상태를 '반환'받음\n        jax_state[:] = [tv, ntv, ov]\n        return loss, predictions\n\ndef sync_jax_state():\n    \"\"\"JAX는 상태를 밖에서 들고 다녔으니, 끝나면 모델 변수에 다시 써 넣어요.\"\"\"\n    if BACKEND == \"jax\":\n        for v, val in zip(model.trainable_variables, jax_state[0]): v.assign(val)\n        for v, val in zip(model.non_trainable_variables, jax_state[1]): v.assign(val)\n\nprint(\"이번 실행의 그레이디언트 방식:\", BACKEND)",
   "out": "이번 실행의 그레이디언트 방식: torch",
   "imgs": []
  },
  {
   "t": "code",
   "src": "batch_size, epochs = 128, 3\nfor epoch in range(epochs):\n    for m in metrics + [loss_tracking_metric]: m.reset_state()           # 에포크마다 지표 초기화\n    perm = np.random.default_rng(epoch).permutation(len(train_images))\n    for i in range(0, len(train_images), batch_size):                     # ① 배치 뽑기\n        idx = perm[i:i + batch_size]\n        x = ops.convert_to_tensor(train_images[idx]); y = ops.convert_to_tensor(train_labels[idx])\n        loss, preds = train_step(x, y)\n        for m in metrics: m.update_state(y, preds)\n        loss_tracking_metric.update_state(loss)\n    print(f\"에포크 {epoch + 1}: loss {float(loss_tracking_metric.result()):.4f} | \"\n          f\"accuracy {float(metrics[0].result()):.4f}\")\n\nsync_jax_state()\nval_preds = model.predict(val_images, verbose=0)\nprint(\"검증 정확도:\", round(float((val_preds.argmax(axis=1) == val_labels).mean()), 4))",
   "out": "에포크 1: loss 0.3459 | accuracy 0.8985\n에포크 2: loss 0.1677 | accuracy 0.9496\n에포크 3: loss 0.1289 | accuracy 0.9620\n검증 정확도: 0.9705",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<p>💬 <b>설명 포인트</b>: 이 루프와 <code>model.fit()</code>은 무엇이 다를까?</p><blockquote><p>계산은 같아요. <code>fit()</code>은 여기에 <b>진행 표시줄, 콜백, 검증, 분산 훈련, 컴파일 최적화</b>(XLA · <code>jit</code>)를 얹어 줄 뿐이에요.<br>직접 짜면 이 모든 걸 직접 챙겨야 해요 → 그래서 보통은 6-4처럼 <b>훈련 스텝만</b> 바꿔요.</p></blockquote><h5>6-4. <code>train_step()</code> 오버라이드 — 스텝만 바꾸고 <code>fit()</code>의 편의는 그대로</h5>"
  },
  {
   "t": "code",
   "src": "loss_fn = keras.losses.SparseCategoricalCrossentropy()\nloss_tracker = keras.metrics.Mean(name=\"loss\")\n\nclass CustomModel(keras.Model):\n    @property\n    def metrics(self):\n        return [loss_tracker]          # 여기 적은 지표는 에포크마다 fit()이 reset_state() 해 줌\n\n    if BACKEND == \"torch\":\n        def train_step(self, data):\n            inputs, targets = data\n            predictions = self(inputs, training=True)\n            loss = loss_fn(targets, predictions)\n            self.zero_grad()\n            loss.backward()\n            gradients = [v.value.grad for v in self.trainable_weights]\n            with torch.no_grad():\n                self.optimizer.apply(gradients, self.trainable_weights)\n            loss_tracker.update_state(loss)\n            return {\"loss\": loss_tracker.result()}\n\n    elif BACKEND == \"tensorflow\":\n        def train_step(self, data):\n            inputs, targets = data\n            with tf.GradientTape() as tape:\n                predictions = self(inputs, training=True)\n                loss = loss_fn(targets, predictions)\n            gradients = tape.gradient(loss, self.trainable_weights)\n            self.optimizer.apply(gradients, self.trainable_weights)\n            loss_tracker.update_state(loss)\n            return {\"loss\": loss_tracker.result()}\n\n    elif BACKEND == \"jax\":\n        def compute_loss_and_updates(self, trainable_variables, non_trainable_variables, inputs, targets):\n            predictions, non_trainable_variables = self.stateless_call(\n                trainable_variables, non_trainable_variables, inputs, training=True)\n            return loss_fn(targets, predictions), non_trainable_variables\n\n        def train_step(self, state, data):       # JAX: 상태를 받고 새 상태를 돌려줌\n            (trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables) = state\n            inputs, targets = data\n            grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)\n            (loss, non_trainable_variables), grads = grad_fn(\n                trainable_variables, non_trainable_variables, inputs, targets)\n            trainable_variables, optimizer_variables = self.optimizer.stateless_apply(\n                optimizer_variables, grads, trainable_variables)\n            new_metrics_vars = loss_tracker.stateless_update_state(metrics_variables, loss)\n            logs = {\"loss\": loss_tracker.stateless_result(new_metrics_vars)}\n            return logs, (trainable_variables, non_trainable_variables, optimizer_variables, new_metrics_vars)\n\ninputs = keras.Input(shape=(28 * 28,))\nfeatures = layers.Dense(512, activation=\"relu\")(inputs)\nfeatures = layers.Dropout(0.5)(features)\noutputs = layers.Dense(10, activation=\"softmax\")(features)\nmodel = CustomModel(inputs, outputs)\nmodel.compile(optimizer=keras.optimizers.RMSprop())      # 손실은 train_step 안에서 직접 계산\nmodel.fit(train_images, train_labels, epochs=3, verbose=2)",
   "out": "Epoch 1/3\n1563/1563 - 12s - 8ms/step - loss: 0.2947\nEpoch 2/3\n1563/1563 - 17s - 11ms/step - loss: 0.1581\nEpoch 3/3\n1563/1563 - 25s - 16ms/step - loss: 0.1313",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<hr><h4>정리 · 스스로 설명해 보기</h4><ol><li>Sequential · 함수형 · 서브클래싱을 언제 고르나? 각각 못 하는 것은? → ______</li><li><code>keras.Input</code>이 돌려주는 객체의 정체는? → ______</li><li>다중 출력 모델의 손실은 어떻게 합쳐지나? → ______</li><li><code>EarlyStopping(patience=1)</code> + <code>ModelCheckpoint(save_best_only=True)</code> 조합의 의미는? → ______</li><li>훈련 루프 5단계와, 백엔드마다 다른 단계는? → ______</li><li>지표가 <code>update_state / result / reset_state</code> 세 메서드를 갖는 이유는? → ______</li></ol>"
  }
 ]
});
