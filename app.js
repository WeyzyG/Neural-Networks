// app.js
// Classifier (CNN) + Autoencoder denoiser training/evaluation UI wiring.

import {
  loadTrainFromFiles,
  loadTestFromFiles,
  splitTrainVal,
  getRandomTestBatch,
  draw28x28ToCanvas,
  makeNoisyCleanPairs
} from './data-loader.js';

const els = {
  inputTrain: document.getElementById('trainCsv'),
  inputTest: document.getElementById('testCsv'),
  btnLoad: document.getElementById('btnLoad'),
  btnTrain: document.getElementById('btnTrain'),
  btnEval: document.getElementById('btnEval'),
  btnTest5: document.getElementById('btnTest5'),
  btnSave: document.getElementById('btnSave'),
  btnLoadModel: document.getElementById('btnLoadModel'),
  btnReset: document.getElementById('btnReset'),
  btnVisor: document.getElementById('btnVisor'),
  // AE controls
  noiseMode: document.getElementById('noiseMode'),
  noiseStd: document.getElementById('noiseStd'),
  noiseProb: document.getElementById('noiseProb'),
  btnBuildAE: document.getElementById('btnBuildAE'),
  btnTrainAE: document.getElementById('btnTrainAE'),
  btnEvalAE: document.getElementById('btnEvalAE'),
  btnTest5AE: document.getElementById('btnTest5AE'),
  btnSaveAE: document.getElementById('btnSaveAE'),
  // Info areas
  dataStatus: document.getElementById('dataStatus'),
  metricsText: document.getElementById('metricsText'),
  modelInfo: document.getElementById('modelInfo'),
  previewStrip: document.getElementById('previewStrip'),
  previewLabels: document.getElementById('previewLabels')
};

let state = {
  files: { train: null, test: null },
  cls: { model: null, train: null, testClean: null, testNoisy: null },
  ae: { model: null }, // autoencoder
  noise: { mode: 'gaussian', std: 0.5, prob: 0.1 }
};

// ---------- UI events ----------
els.inputTrain.addEventListener('change', e => state.files.train = e.target.files[0]);
els.inputTest.addEventListener('change', e => state.files.test = e.target.files[0]);
els.btnVisor.addEventListener('click', () => tfvis.visor().toggle());

els.btnLoad.addEventListener('click', onLoadData);
els.btnTrain.addEventListener('click', onTrainClassifier);
els.btnEval.addEventListener('click', onEvaluateClassifier);
els.btnTest5.addEventListener('click', onTest5Classifier);
els.btnSave.addEventListener('click', () => state.cls.model?.save('downloads://mnist-cnn'));

els.btnLoadModel.addEventListener('click', onLoadClassifierFiles);
els.btnReset.addEventListener('click', () => onReset(false));

els.noiseMode.addEventListener('change', async () => { state.noise.mode = els.noiseMode.value; await rebuildNoisyTest(); });
els.noiseStd.addEventListener('input', async () => { state.noise.std = parseFloat(els.noiseStd.value); if (state.noise.mode === 'gaussian') await rebuildNoisyTest(); });
els.noiseProb.addEventListener('input', async () => { state.noise.prob = parseFloat(els.noiseProb.value); if (state.noise.mode === 'sp') await rebuildNoisyTest(); });

// AE
els.btnBuildAE.addEventListener('click', buildAE);
els.btnTrainAE.addEventListener('click', onTrainAE);
els.btnEvalAE.addEventListener('click', onEvalAE);
els.btnTest5AE.addEventListener('click', onTest5AE);
els.btnSaveAE.addEventListener('click', () => state.ae.model?.save('downloads://mnist-ae'));

// ---------- Data load ----------
async function onLoadData() {
  try {
    if (!state.files.train || !state.files.test) {
      alert('Select both train and test CSV files.');
      return;
    }
    await onReset(true);

    state.cls.train = await loadTrainFromFiles(state.files.train);
    state.cls.testClean = await loadTestFromFiles(state.files.test, { noise: 'none' });
    await rebuildNoisyTest();

    els.dataStatus.textContent = `Train: ${state.cls.train.xs.shape[0]} | Test: ${state.cls.testClean.xs.shape[0]} (noisy mode=${state.noise.mode})`;
  } catch (e) {
    console.error(e);
    alert('Load error: ' + e.message);
  }
}

async function rebuildNoisyTest() {
  if (!state.files.test) return;
  // dispose previous
  state.cls.testNoisy?.xs?.dispose();
  state.cls.testNoisy?.ys?.dispose();
  state.cls.testNoisy = await loadTestFromFiles(state.files.test, {
    noise: state.noise.mode,
    stdDev: state.noise.std,
    prob: state.noise.prob
  });
}

// ---------- Classifier ----------
function buildClassifier() {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same', inputShape: [28, 28, 1] }));
  m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  m.add(tf.layers.dropout({ rate: 0.25 }));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  m.add(tf.layers.dropout({ rate: 0.5 }));
  m.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  m.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return m;
}

async function onTrainClassifier() {
  try {
    if (!state.cls.train) return alert('Load data first.');
    state.cls.model?.dispose();
    state.cls.model = buildClassifier();

    const { trainXs, trainYs, valXs, valYs } = splitTrainVal(state.cls.train.xs, state.cls.train.ys, 0.1);
    const callbacks = tfvis.show.fitCallbacks({ name: 'Classifier Training', tab: 'Training Logs' }, ['loss', 'val_loss', 'acc', 'val_acc']);
    const t0 = performance.now();
    await state.cls.model.fit(trainXs, trainYs, {
      epochs: 8, batchSize: 128, shuffle: true, validationData: [valXs, valYs], callbacks
    });
    const dur = ((performance.now() - t0) / 1000).toFixed(1);
    els.metricsText.textContent = `Classifier trained in ${dur}s`;

    trainXs.dispose(); trainYs.dispose(); valXs.dispose(); valYs.dispose();
    renderModelSummary(state.cls.model, 'Classifier');
  } catch (e) {
    console.error(e);
    alert('Train error: ' + e.message);
  }
}

async function onEvaluateClassifier() {
  try {
    if (!state.cls.model || !state.cls.testClean) return alert('Need model and data.');
    const use = state.cls.testNoisy ?? state.cls.testClean;
    const res = await state.cls.model.evaluate(use.xs, use.ys, { batchSize: 256 });
    const acc = Array.isArray(res) ? res[1].dataSync()[0] : res.dataSync()[0];
    els.metricsText.textContent = `Classifier test accuracy (noisy=${state.noise.mode}) = ${(acc * 100).toFixed(2)}%`;

    const preds = state.cls.model.predict(use.xs).argMax(-1);
    const labels = use.ys.argMax(-1);
    const cm = await tfvis.metrics.confusionMatrix(labels, preds, 10);
    await tfvis.render.confusionMatrix({ name: 'Confusion Matrix', tab: 'Metrics' }, { values: cm, tickLabels: [...Array(10)].map((_, i) => String(i)) });
    const pca = tfvis.metrics.perClassAccuracy(labels, preds);
    await tfvis.render.barchart({ name: 'Per‑class Accuracy', tab: 'Metrics' }, Object.keys(pca).map(k => ({ x: k, y: pca[k] })));
    preds.dispose(); labels.dispose();
  } catch (e) {
    console.error(e);
    alert('Eval error: ' + e.message);
  }
}

async function onTest5Classifier() {
  try {
    if (!state.cls.model || !state.cls.testClean) return alert('Need model and data.');
    const use = state.cls.testNoisy ?? state.cls.testClean;
    els.previewStrip.innerHTML = ''; els.previewLabels.innerHTML = '';
    const { batchXs, batchYs } = getRandomTestBatch(use.xs, use.ys, 5);
    const preds = state.cls.model.predict(batchXs).argMax(-1);
    const yTrue = batchYs.argMax(-1);
    const pa = Array.from(preds.dataSync()); const ya = Array.from(yTrue.dataSync());
    for (const img of tf.unstack(batchXs)) {
      const c = document.createElement('canvas'); els.previewStrip.appendChild(c);
      draw28x28ToCanvas(img, c, 4); img.dispose();
    }
    for (let i = 0; i < pa.length; i++) {
      const s = document.createElement('span'); s.textContent = String(pa[i]);
      s.style.color = pa[i] === ya[i] ? '#2e7d32' : '#c0392b'; els.previewLabels.appendChild(s);
    }
    preds.dispose(); yTrue.dispose(); batchXs.dispose(); batchYs.dispose();
  } catch (e) {
    console.error(e);
    alert('Preview error: ' + e.message);
  }
}

async function onLoadClassifierFiles() {
  try {
    const j = document.getElementById('modelJson').files[0];
    const b = document.getElementById('modelBin').files[0];
    if (!j || !b) return alert('Select model.json and weights.bin');
    state.cls.model?.dispose();
    state.cls.model = await tf.loadLayersModel(tf.io.browserFiles([j, b]));
    renderModelSummary(state.cls.model, 'Classifier (loaded)');
  } catch (e) {
    console.error(e);
    alert('Load model error: ' + e.message);
  }
}

// ---------- Autoencoder (Steps 2–4) ----------
function buildAE() {
  state.ae.model?.dispose();
  // Simple conv autoencoder: 28x28x1 -> bottleneck -> 28x28x1
  const m = tf.sequential();
  // Encoder
  m.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same', inputShape: [28, 28, 1] }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' }));
  m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' })); // 7x7
  // Decoder
  m.add(tf.layers.conv2dTranspose({ filters: 64, kernelSize: 3, strides: 2, activation: 'relu', padding: 'same' })); // 14x14
  m.add(tf.layers.conv2dTranspose({ filters: 32, kernelSize: 3, strides: 2, activation: 'relu', padding: 'same' })); // 28x28
  m.add(tf.layers.conv2d({ filters: 1, kernelSize: 3, activation: 'sigmoid', padding: 'same' })); // output [0,1]
  m.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
  state.ae.model = m;
  renderModelSummary(state.ae.model, 'Autoencoder');
}

async function onTrainAE() {
  try {
    if (!state.cls.train) return alert('Load data first.');
    if (!state.ae.model) buildAE();

    // Build noisy-clean pairs from train.xs
    const { noisy, clean } = makeNoisyCleanPairs(state.cls.train.xs, state.noise.mode, { stdDev: state.noise.std, prob: state.noise.prob });
    const { trainXs, valXs } = splitTrainVal(noisy, clean, 0.1); // reuse split for tensors of same first dim
    const { trainXs: trainClean, valXs: valClean } = splitTrainVal(clean, clean, 0.1);

    const callbacks = tfvis.show.fitCallbacks({ name: 'AE Training', tab: 'Training Logs' }, ['loss', 'val_loss']);
    const t0 = performance.now();
    await state.ae.model.fit(trainXs, trainClean, {
      epochs: 8, batchSize: 128, shuffle: true, validationData: [valXs, valClean], callbacks
    });
    const dur = ((performance.now() - t0) / 1000).toFixed(1);
    els.metricsText.textContent = `AE trained in ${dur}s`;

    // dispose temp
    trainXs.dispose(); valXs.dispose(); trainClean.dispose(); valClean.dispose(); noisy.dispose(); // clean is train.xs reference; don't dispose
  } catch (e) {
    console.error(e);
    alert('AE train error: ' + e.message);
  }
}

async function onEvalAE() {
  try {
    if (!state.ae.model || !state.cls.testClean) return alert('Need AE and data.');
    const noisy = await getNoisyTestXs();
    const res = await state.ae.model.evaluate(noisy, state.cls.testClean.xs, { batchSize: 256 });
    const loss = Array.isArray(res) ? res[0].dataSync()[0] : res.dataSync()[0];
    els.metricsText.textContent = `AE MSE on noisy→clean: ${loss.toFixed(5)}`;
    noisy.dispose();
  } catch (e) {
    console.error(e);
    alert('AE eval error: ' + e.message);
  }
}

async function onTest5AE() {
  try {
    if (!state.ae.model || !state.cls.testClean) return alert('Need AE and data.');
    els.previewStrip.innerHTML = ''; els.previewLabels.innerHTML = '';
    const noisy = await getNoisyTestXs();
    const { batchXs } = getRandomTestBatch(noisy, null, 5);
    const denoised = state.ae.model.predict(batchXs);

    // Render pairs: noisy (top row) then denoised (bottom row)
    const k = batchXs.shape[0];
    for (let i = 0; i < k; i++) {
      const c = document.createElement('canvas'); els.previewStrip.appendChild(c);
      draw28x28ToCanvas(batchXs.slice([i, 0, 0, 0], [1, 28, 28, 1]).squeeze(), c, 4);
    }
    const br = document.createElement('div'); br.style.flexBasis = '100%'; els.previewStrip.appendChild(br);
    for (let i = 0; i < k; i++) {
      const c = document.createElement('canvas'); els.previewStrip.appendChild(c);
      draw28x28ToCanvas(denoised.slice([i, 0, 0, 0], [1, 28, 28, 1]).squeeze(), c, 4);
    }
    els.previewLabels.textContent = 'Top: noisy, Bottom: denoised';

    batchXs.dispose(); denoised.dispose(); noisy.dispose();
  } catch (e) {
    console.error(e);
    alert('AE preview error: ' + e.message);
  }
}

async function getNoisyTestXs() {
  // regenerate from clean each time to avoid compounding
  const { noisy } = makeNoisyCleanPairs(state.cls.testClean.xs, state.noise.mode, { stdDev: state.noise.std, prob: state.noise.prob });
  return noisy;
}

// ---------- Common ----------
function renderModelSummary(model, title) {
  els.modelInfo.innerHTML = '';
  const div = document.createElement('div');
  div.textContent = `${title} — layers: ${model.layers.length}, params: ${model.countParams()}`;
  els.modelInfo.appendChild(div);
  model.summary(undefined, undefined, line => {
    const el = document.createElement('div'); el.className = 'muted'; el.textContent = line; els.modelInfo.appendChild(el);
  });
}

async function onReset(keepFiles = false) {
  try {
    els.metricsText.textContent = 'No metrics yet';
    els.previewStrip.innerHTML = ''; els.previewLabels.innerHTML = '';
    els.modelInfo.textContent = 'No model loaded';
    state.cls.model?.dispose(); state.cls.model = null;
    state.ae.model?.dispose(); state.ae.model = null;
    for (const k of ['train', 'testClean', 'testNoisy']) {
      if (state.cls[k]?.xs) state.cls[k].xs.dispose();
      if (state.cls[k]?.ys) state.cls[k].ys.dispose();
      state.cls[k] = null;
    }
    if (!keepFiles) {
      els.inputTrain.value = ''; els.inputTest.value = '';
      state.files.train = null; state.files.test = null;
    }
  } catch (e) { console.error(e); }
}
