/* 자동 생성: scratchpad build_nb.py --run 으로 노트북을 실행한 기록 */
BOOK.addLab({
 "ch": 3,
 "title": "3장 실습 · 세 프레임워크로 같은 분류기 만들기",
 "file": "projects/ch03-frameworks/ch03_frameworks.ipynb",
 "run": "2026-11-14 · JAX 백엔드 · CPU",
 "cells": [
  {
   "t": "md",
   "html": "<h3>3장 · 텐서플로, 파이토치, JAX 그리고 케라스</h3><p>『케라스 창시자에게 배우는 딥러닝 개정 3판』 3장을 바로 돌려 보는 노트북이에요.<br>위에서부터 차례로 실행하면 돼요.</p><div class=\"lab-tw\"><table><thead><tr><th>절</th><th>내용</th><th>내 필기</th></tr></thead><tbody><tr><td>0</td><td>케라스 백엔드 고르기</td><td>케라스</td></tr><tr><td>1</td><td>텐서플로: 텐서 · 변수 · <code>GradientTape</code> · 선형 분류기</td><td>tf-torch-jax, 그레이디언트 구하기</td></tr><tr><td>2</td><td>파이토치: <code>requires_grad</code> · <code>backward()</code> · <code>.grad</code> <b>누적</b></td><td>그레이디언트 구하기</td></tr><tr><td>3</td><td>JAX: <code>jnp</code> · <code>jax.grad</code> · 상태 비저장</td><td>그레이디언트 구하기</td></tr><tr><td>4</td><td>케라스: 층과 모델</td><td>층과 모델</td></tr><tr><td>5</td><td>케라스: <code>compile → fit → evaluate → predict</code></td><td>설정-훈련-평가-사용</td></tr></tbody></table></div>"
  },
  {
   "t": "md",
   "html": "<h4>0. 케라스 백엔드 고르기 (§3.6)</h4><p>케라스는 고수준이라 혼자 못 돌아요. 저수준 <b>백엔드</b>(텐서플로 · 파이토치 · JAX) 위에서 돌아요.</p><ul><li><code>KERAS_BACKEND</code>는 <b><code>import keras</code> 전에</b> 정해야 해요.</li><li>아무것도 안 정하면 기본은 텐서플로예요.</li><li>백엔드를 바꾸려면 커널을 재시작한 뒤 이 셀부터 다시 실행해요.</li></ul>"
  },
  {
   "t": "code",
   "src": "import os\nos.environ[\"KERAS_BACKEND\"] = \"jax\"   # \"tensorflow\" / \"torch\" / \"jax\" 중 골라 보기\n\nimport numpy as np\nimport matplotlib.pyplot as plt",
   "out": "",
   "imgs": []
  },
  {
   "t": "code",
   "src": "plt.rcParams[\"font.family\"] = \"Malgun Gothic\"     # 그래프 한글 깨짐 방지 (윈도)\nplt.rcParams[\"axes.unicode_minus\"] = False",
   "out": "",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h5>공통 실습 데이터: 점 구름 2개 (§3.3)</h5><p>세 프레임워크로 <b>같은 선형 분류기</b>를 만들어서 비교해 볼 거예요.</p>"
  },
  {
   "t": "code",
   "src": "np.random.seed(42)            # 실행할 때마다 같은 데이터\nnum_samples_per_class = 1000\nnegative_samples = np.random.multivariate_normal(\n    mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class)\npositive_samples = np.random.multivariate_normal(\n    mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class)\n\ninputs = np.vstack((negative_samples, positive_samples)).astype(\"float32\")\ntargets = np.vstack((np.zeros((num_samples_per_class, 1), dtype=\"float32\"),\n                     np.ones((num_samples_per_class, 1), dtype=\"float32\")))\n\nplt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0], s=4)\nplt.title(\"클래스 0 / 1 점 구름\")\nplt.show()",
   "out": "",
   "imgs": [
    "lab/ch03/fig01.png"
   ]
  },
  {
   "t": "md",
   "html": "<h4>1. 텐서플로 (§3.3)</h4><ul><li>텐서는 <b>상수</b>라서 값 할당 불가 → 바뀌는 값은 <code>tf.Variable</code></li><li><code>GradientTape</code> 블록 안의 연산을 <b>테이프로 기록</b> → <code>tape.gradient()</code></li></ul>"
  },
  {
   "t": "code",
   "src": "import tensorflow as tf\ntf.random.set_seed(42)\n\nx = tf.ones(shape=(2, 1))\nprint(x)\n\nv = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))\nv[0, 0].assign(3.)          # 변수는 assign으로 바꿈\nv.assign_add(tf.ones((3, 1)))\nprint(v)",
   "out": "tf.Tensor(\n[[1.]\n [1.]], shape=(2, 1), dtype=float32)\n<tf.Variable 'Variable:0' shape=(3, 1) dtype=float32, numpy=\narray([[4.       ],\n       [0.1573742],\n       [1.3194337]], dtype=float32)>",
   "imgs": []
  },
  {
   "t": "code",
   "src": "# 그레이디언트: y = x² 를 x = 3 에서 미분 → 6\ninput_var = tf.Variable(initial_value=3.0)\nwith tf.GradientTape() as tape:\n    result = tf.square(input_var)\ngradient = tape.gradient(result, input_var)\nprint(gradient)",
   "out": "tf.Tensor(6.0, shape=(), dtype=float32)",
   "imgs": []
  },
  {
   "t": "code",
   "src": "# 선형 분류기: prediction = W·x + b, 손실 = MSE\ninput_dim, output_dim = 2, 1\nW = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))\nb = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))\n\ndef model(inputs, W, b):\n    return tf.matmul(inputs, W) + b\n\ndef mean_squared_error(targets, predictions):\n    return tf.reduce_mean(tf.square(targets - predictions))\n\nlearning_rate = 0.1\n\n@tf.function(jit_compile=True)   # 그래프 모드 + XLA 컴파일 → 빨라짐\ndef training_step(inputs, targets, W, b):\n    with tf.GradientTape() as tape:\n        predictions = model(inputs, W, b)\n        loss = mean_squared_error(targets, predictions)\n    grad_W, grad_b = tape.gradient(loss, [W, b])\n    W.assign_sub(grad_W * learning_rate)\n    b.assign_sub(grad_b * learning_rate)\n    return loss\n\nfor step in range(40):\n    loss = training_step(inputs, targets, W, b)\n    if step % 10 == 0 or step == 39:\n        print(f\"스텝 {step}: 손실 {float(loss):.4f}\")",
   "out": "스텝 0: 손실 2.5915\n스텝 10: 손실 0.0773\n스텝 20: 손실 0.0464\n스텝 30: 손실 0.0338\n스텝 39: 손실 0.0289",
   "imgs": []
  },
  {
   "t": "code",
   "src": "predictions = model(inputs, W, b)\nx_line = np.linspace(-1, 4, 100)\ny_line = -W[0, 0].numpy() / W[1, 0].numpy() * x_line + (0.5 - b[0].numpy()) / W[1, 0].numpy()\nplt.plot(x_line, y_line, \"-r\")\nplt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5, s=4)\nplt.title(\"텐서플로 선형 분류기 (빨간 선 = 결정 경계)\")\nplt.show()",
   "out": "",
   "imgs": [
    "lab/ch03/fig02.png"
   ]
  },
  {
   "t": "md",
   "html": "<h4>2. 파이토치 (§3.4)</h4><ul><li><code>requires_grad=True</code> 텐서 → <code>loss.backward()</code> → 결과는 <code>.grad</code>에 저장 (<b>텐서가 거꾸로 추적</b>)</li><li>⚠️ 내 필기에 빠졌던 것: <code>.grad</code>는 <code>backward()</code>할 때마다 <b>누적</b>돼요</li></ul>"
  },
  {
   "t": "code",
   "src": "import torch\ntorch.manual_seed(42)\n\ninput_var = torch.tensor(3.0, requires_grad=True)\nresult = torch.square(input_var)\nresult.backward()\nprint(\"1번째:\", input_var.grad)    # 6\n\nresult = torch.square(input_var)\nresult.backward()\nprint(\"2번째:\", input_var.grad)    # 12 ← 누적됨!\n\ninput_var.grad = None              # 그래서 스텝마다 비워 줘야 함",
   "out": "1번째: tensor(6.)\n2번째: tensor(12.)",
   "imgs": []
  },
  {
   "t": "code",
   "src": "W = torch.rand(input_dim, output_dim, requires_grad=True)\nb = torch.zeros(output_dim, requires_grad=True)\nt_inputs, t_targets = torch.tensor(inputs), torch.tensor(targets)\n\ndef model(inputs, W, b):\n    return torch.matmul(inputs, W) + b\n\ndef mean_squared_error(targets, predictions):\n    return torch.mean(torch.square(targets - predictions))\n\ndef training_step(inputs, targets, W, b):\n    predictions = model(inputs, W, b)\n    loss = mean_squared_error(targets, predictions)\n    loss.backward()\n    with torch.no_grad():          # 업데이트는 역전파에 포함 X\n        W -= W.grad * learning_rate\n        b -= b.grad * learning_rate\n    W.grad = None                  # 누적 방지\n    b.grad = None\n    return loss\n\nfor step in range(40):\n    loss = training_step(t_inputs, t_targets, W, b)\n    if step % 10 == 0 or step == 39:\n        print(f\"스텝 {step}: 손실 {loss.item():.4f}\")",
   "out": "스텝 0: 손실 7.3616\n스텝 10: 손실 0.1006\n스텝 20: 손실 0.0560\n스텝 30: 손실 0.0377\n스텝 39: 손실 0.0307",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>3. JAX (§3.5)</h4><ul><li>넘파이와 <b>같은 API</b> (<code>jax.numpy</code> = <code>jnp</code>)</li><li><b>함수형 · 상태 비저장</b>: 손실 <i>함수</i>를 <code>jax.grad</code>에 넣으면 그레이디언트 <i>함수</i>가 나와요 (f → f′)</li><li>상태(W, b)는 함수 밖에서 들고 다니고, 새 값을 <b>반환</b>받아요</li></ul>"
  },
  {
   "t": "code",
   "src": "import jax\nimport jax.numpy as jnp\n\ndef square(x):\n    return x ** 2\n\ngrad_square = jax.grad(square)     # 함수 → 도함수\nprint(grad_square(3.0))            # 6",
   "out": "6.0",
   "imgs": []
  },
  {
   "t": "code",
   "src": "def model(inputs, W, b):\n    return jnp.matmul(inputs, W) + b\n\ndef mean_squared_error(targets, predictions):\n    return jnp.mean(jnp.square(targets - predictions))\n\ndef compute_loss(state, inputs, targets):\n    W, b = state\n    return mean_squared_error(targets, model(inputs, W, b))\n\ngrad_fn = jax.value_and_grad(compute_loss)   # 손실과 그레이디언트를 같이\n\n@jax.jit\ndef training_step(inputs, targets, W, b):\n    loss, (grad_W, grad_b) = grad_fn((W, b), inputs, targets)\n    W = W - grad_W * learning_rate   # 제자리 수정 X → 새 값을 반환\n    b = b - grad_b * learning_rate\n    return loss, W, b\n\nW = jax.random.uniform(jax.random.key(42), shape=(input_dim, output_dim))\nb = jnp.zeros(shape=(output_dim,))\nfor step in range(40):\n    loss, W, b = training_step(inputs, targets, W, b)\n    if step % 10 == 0 or step == 39:\n        print(f\"스텝 {step}: 손실 {float(loss):.4f}\")",
   "out": "스텝 0: 손실 3.1345\n스텝 10: 손실 0.0774\n스텝 20: 손실 0.0465\n스텝 30: 손실 0.0338\n스텝 39: 손실 0.0289",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>4. 케라스: 층과 모델 (§3.6.2)</h4><ul><li><b>층</b> = 상태(가중치) + 연산(<code>call</code>)</li><li><code>build()</code>는 첫 입력 크기를 보고 가중치를 만들어요 (자동 크기 추론)</li><li><b>모델</b> = 층의 그래프. 모델 구조가 <b>가설 공간을 정의</b>해요</li></ul>"
  },
  {
   "t": "code",
   "src": "import keras\nkeras.utils.set_random_seed(42)\nprint(\"백엔드:\", keras.backend.backend())\n\nclass SimpleDense(keras.Layer):\n    def __init__(self, units, activation=None):\n        super().__init__()\n        self.units = units\n        self.activation = activation\n\n    def build(self, input_shape):          # 첫 호출 때 자동 실행\n        batch_dim, input_dim = input_shape\n        self.W = self.add_weight(shape=(input_dim, self.units), initializer=\"random_normal\")\n        self.b = self.add_weight(shape=(self.units,), initializer=\"zeros\")\n\n    def call(self, inputs):                 # relu(x·W + b)\n        y = keras.ops.matmul(inputs, self.W) + self.b\n        if self.activation is not None:\n            y = self.activation(y)\n        return y\n\nmy_dense = SimpleDense(units=32, activation=keras.ops.relu)\noutput = my_dense(keras.ops.ones(shape=(2, 784)))\nprint(\"출력 크기:\", output.shape)\nprint(\"가중치:\", [w.shape for w in my_dense.weights])",
   "out": "백엔드: jax\n출력 크기: (2, 32)\n가중치: [(784, 32), (32,)]",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>5. 케라스: compile → fit → evaluate → predict (§3.6.4)</h4><p>설정 → 훈련 → 평가 → 사용. 앞에서 세 번 손으로 짠 선형 분류기를 케라스로 다시 만들어요.</p>"
  },
  {
   "t": "code",
   "src": "# 섞고 검증 데이터 30% 떼기 (validation_data = 훈련에 안 쓰고 확인용)\nindices = np.random.permutation(len(inputs))\nshuffled_inputs, shuffled_targets = inputs[indices], targets[indices]\nn_val = int(0.3 * len(inputs))\nval_inputs, val_targets = shuffled_inputs[:n_val], shuffled_targets[:n_val]\ntrain_inputs, train_targets = shuffled_inputs[n_val:], shuffled_targets[n_val:]\n\nmodel = keras.Sequential([keras.layers.Dense(1)])\n\nmodel.compile(                                   # 설정\n    optimizer=keras.optimizers.RMSprop(learning_rate=0.1),\n    loss=keras.losses.MeanSquaredError(),        # 손실 고르기가 제일 중요 (4장)\n    metrics=[keras.metrics.BinaryAccuracy()],\n)\n\nhistory = model.fit(                             # 훈련\n    train_inputs, train_targets,\n    epochs=5, batch_size=16,\n    validation_data=(val_inputs, val_targets),\n    verbose=2,                                   # 에포크마다 한 줄씩\n)",
   "out": "Epoch 1/5\n88/88 - 1s - 6ms/step - binary_accuracy: 0.9071 - loss: 0.2966 - val_binary_accuracy: 0.9950 - val_loss: 0.0610\nEpoch 2/5\n88/88 - 0s - 497us/step - binary_accuracy: 0.9529 - loss: 0.0759 - val_binary_accuracy: 1.0000 - val_loss: 0.0462\nEpoch 3/5\n88/88 - 0s - 558us/step - binary_accuracy: 0.9614 - loss: 0.0694 - val_binary_accuracy: 0.9933 - val_loss: 0.0462\nEpoch 4/5\n88/88 - 0s - 486us/step - binary_accuracy: 0.9529 - loss: 0.0710 - val_binary_accuracy: 1.0000 - val_loss: 0.0455\nEpoch 5/5\n88/88 - 0s - 425us/step - binary_accuracy: 0.9686 - loss: 0.0714 - val_binary_accuracy: 1.0000 - val_loss: 0.0365",
   "imgs": []
  },
  {
   "t": "code",
   "src": "print(history.history.keys())    # 에포크별 손실 · 지표가 쌓여 있음\nplt.plot(history.history[\"loss\"], label=\"훈련 손실\")\nplt.plot(history.history[\"val_loss\"], label=\"검증 손실\")\nplt.xlabel(\"에포크\"); plt.legend(); plt.show()",
   "out": "dict_keys(['binary_accuracy', 'loss', 'val_binary_accuracy', 'val_loss'])",
   "imgs": [
    "lab/ch03/fig03.png"
   ]
  },
  {
   "t": "code",
   "src": "loss_and_metrics = model.evaluate(val_inputs, val_targets, batch_size=128, verbose=0)   # 평가\nprint(\"검증 손실, 정확도:\", loss_and_metrics)\n\npredictions = model.predict(val_inputs, batch_size=128, verbose=0)                    # 사용\nprint(predictions[:5])",
   "out": "검증 손실, 정확도: [0.036530233919620514, 1.0]\n[[ 0.67477316]\n [-0.4138842 ]\n [ 1.1705348 ]\n [-0.11963329]\n [ 0.6468539 ]]",
   "imgs": []
  },
  {
   "t": "md",
   "html": "<h4>정리</h4><div class=\"lab-tw\"><table><thead><tr><th></th><th>그레이디언트</th><th>상태</th></tr></thead><tbody><tr><td>텐서플로</td><td><code>GradientTape</code>로 <b>기록</b></td><td><code>tf.Variable</code> + <code>assign</code></td></tr><tr><td>파이토치</td><td>텐서가 <b>역추적</b>, <code>.grad</code>에 누적</td><td><code>requires_grad</code> 텐서, 직접 비우기</td></tr><tr><td>JAX</td><td><code>jax.grad</code>로 <b>함수 변환</b></td><td>없음 → 반환값으로 들고 다님</td></tr><tr><td>케라스</td><td>위 셋 중 하나를 백엔드로</td><td><code>compile → fit → evaluate → predict</code></td></tr></tbody></table></div><p class=\"pen\">*결국, 길만 다르고 답은 같다. 케라스면 갈아타기 쉽다.</p><p>👉 해 볼 것: 0번 셀에서 백엔드를 <code>\"torch\"</code>나 <code>\"tensorflow\"</code>로 바꾸고 커널 재시작 → 4 · 5절이 그대로 도는지 확인하기</p>"
  }
 ]
});
