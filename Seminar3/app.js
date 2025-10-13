// app.js
// Wire UI + training/eval logic. Adds Step 1 toggle to use noisy test data.

import {
  loadTrainFromFiles,
  loadTestFromFiles,
  splitTrainVal,
  getRandomTestBatch,
  draw28x28ToCanvas,
  addGaussianNoise01,
  addSaltPepper01
} from './data-loader.js';

let state = {
  trainFile: null,
  testFile: null,
  train: null,
  testClean: null, // {xs, ys}
  testNoisy: null, // cached noisy copy per current noise settings
  model: null,
  noiseMode: 'gaussian', // 'none'|'gaussian'|'sp'
  noiseStd: 0.5,
  noiseProb: 0.1
};

// ---- UI bindings ----
const els = {
  inputTrain: document.getElementById('trainCsv'),
  inputTest: document.getElementById('testCsv'),
  btnLoad: document.getElementById('btnLoad'),
  btnTrain: document.getElementById('btnTrain'),
  btnEvaluate: document.getElementById('btnEval'),
  btnTest5: document.getElementById('btnTest5'),
  btnSave: document.getElementById('btnSave'),
  btnLoadModel: document.getElementById('btnLoadModel'),
  btnReset: document.getElementById('btnReset'),
  visorToggle: document.getElementById('btnVisor'),
  // Step 1 UI
  noiseMode: document.getElementById('noiseMode'),
  noiseStd: document.getElementById('noiseStd'),
  noiseProb: document.getElementById('noiseProb'),
  dataStatus: document.getElementById('dataStatus'),
  previewRow: document.getElementById('previewRow'),
  previewLabels: document.getElementById('previewLabels'),
  metricsText: document.getElementById('metricsText'),
  modelInfo: document.getElementById('modelInfo')
};

// Attach events for file inputs
els.inputTrain.addEventListener('change', e => state.trainFile = e.target.files[0]);
els.inputTest.addEventListener('change', e => state.testFile = e.target.files[0]);

// Step 1: controls for noise
els.noiseMode.addEventListener('change', async () => {
  state.noiseMode = els.noiseMode.value;
  await rebuildNoisyCache();
});
els.noiseStd.addEventListener('input', async () => {
  state.noiseStd = parseFloat(els.noiseStd.value);
  if (state.noiseMode === 'gaussian') await rebuildNoisyCache();
});
els.noiseProb.addEventListener('input', async () => {
  state.noiseProb = parseFloat(els.noiseProb.value);
  if (state.noiseMode === 'sp') await rebuildNoisyCache();
});

// Main buttons
els.btnLoad.addEventListener('click', onLoadData);
els.btnTrain.addEventListener('click', onTrain);
els.btnEvaluate.addEventListener('click', onEvaluate);
els.btnTest5.addEventListener('click', onTestFive);
els.btnSave.addEventListener('click', async () => state.model && await state.model.save('downloads://mnist-cnn'));
els.btnLoadModel.addEventListener('click', onLoadFromFiles);
els.btnReset.addEventListener('click', onReset);
els.visorToggle.addEventListener('click', () => tfvis.visor().toggle());

// ---- Data load ----
async function onLoadData() {
  try {
    if (!state.trainFile || !state.testFile) {
      alert('Please select both Train and Test CSV files.');
      return;
    }
    // Dispose previous
    await onReset({ keepFiles: true });

    const train = await loadTrainFromFiles(state.trainFile);
    const testClean = await loadTestFromFiles(state.testFile, { noise: 'none' });
    state.train = train;
    state.testClean = testClean;

    await rebuildNoisyCache();

    els.dataStatus.textContent =
      `Train: ${train.xs.shape[0]} | Test: ${testClean.xs.shape[0]} (noisy: ${state.testNoisy?.xs.shape[0]})`;
  } catch (e) {
    console.error(e);
    alert('Failed to load data: ' + e.message);
  }
}

async function rebuildNoisyCache() {
  if (!state.testClean) return;
  // regenerate noisy cache from clean to avoid compounding noise
  state.testNoisy?.xs?.dispose();
  state.testNoisy?.ys?.dispose();
  const opt = state.noiseMode === 'gaussian'
    ? { noise: 'gaussian', stdDev: state.noiseStd }
    : state.noiseMode === 'sp'
      ? { noise: 'sp', prob: state.noiseProb }
      : { noise: 'none' };
  const reloaded = await loadTestFromFiles(state.testFile, opt);
  state.testNoisy = reloaded;
}

// ---- Model ----
function buildModel() {
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

// ---- Train ----
async function onTrain() {
  try {
    if (!state.train) return alert('Load data first');
    if (state.model) state.model.dispose();
    state.model = buildModel();

    const { trainXs, trainYs, valXs, valYs } =
      splitTrainVal(state.train.xs, state.train.ys, 0.1);

    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const surface = { name: 'Training', tab: 'Training Logs' };
    const callbacks = tfvis.show.fitCallbacks(surface, metrics);

    const start = performance.now();
    await state.model.fit(trainXs, trainYs, {
      epochs: 8,
      batchSize: 128,
      shuffle: true,
      validationData: [valXs, valYs],
      callbacks
    });
    const dur = ((performance.now() - start) / 1000).toFixed(1);
    els.metricsText.textContent = `Training finished in ${dur}s`;
    trainXs.dispose(); trainYs.dispose(); valXs.dispose(); valYs.dispose();

    state.model.summary(undefined, undefined, x => {
      const el = document.createElement('div'); el.textContent = x;
      els.modelInfo.appendChild(el);
    });
  } catch (e) {
    console.error(e);
    alert('Training error: ' + e.message);
  }
}

// ---- Evaluate ----
async function onEvaluate() {
  try {
    if (!state.model || !state.testClean) return alert('Need model and data');
    // Evaluate on noisy test data per Step 1
    const use = state.testNoisy ?? state.testClean;
    const evalRes = await state.model.evaluate(use.xs, use.ys, { batchSize: 256 });
    const acc = Array.isArray(evalRes) ? evalRes[1].dataSync()[0] : evalRes.dataSync()[0];
    els.metricsText.textContent = `Test accuracy (noisy=${state.noiseMode}): ${(acc * 100).toFixed(2)}%`;

    // Confusion matrix
    const preds = state.model.predict(use.xs).argMax(-1);
    const labels = use.ys.argMax(-1);
    const cm = await tfvis.metrics.confusionMatrix(labels, preds, 10);
    await tfvis.render.confusionMatrix(
      { name: 'Confusion Matrix', tab: 'Metrics' },
      { values: cm, tickLabels: [...Array(10)].map((_, i) => String(i)) }
    );
    // Per-class accuracy
    const classAcc = tfvis.metrics.perClassAccuracy(labels, preds);
    const series = Object.keys(classAcc).map(k => ({ index: Number(k), acc: classAcc[k] }));
    await tfvis.render.barchart(
      { name: 'Per-class Accuracy', tab: 'Metrics' },
      series.map(s => ({ x: String(s.index), y: s.acc }))
    );
    preds.dispose(); labels.dispose();
  } catch (e) {
    console.error(e);
    alert('Evaluation error: ' + e.message);
  }
}

// ---- Test 5 Random ----
async function onTestFive() {
  try {
    if (!state.model || !state.testClean) return alert('Need model and data');
    const use = state.testNoisy ?? state.testClean;

    // Prepare UI row
    els.previewRow.innerHTML = '';
    els.previewLabels.innerHTML = '';

    const { batchXs, batchYs } = getRandomTestBatch(use.xs, use.ys, 5);
    const preds = state.model.predict(batchXs).argMax(-1);
    const labels = batchYs.argMax(-1);

    const predArr = Array.from(preds.dataSync());
    const labArr = Array.from(labels.dataSync());

    const xsSplit = tf.unstack(batchXs);
    for (let i = 0; i < xsSplit.length; i++) {
      const canvas = document.createElement('canvas');
      canvas.style.marginRight = '8px';
      els.previewRow.appendChild(canvas);
      draw28x28ToCanvas(xsSplit[i], canvas, 4);

      const span = document.createElement('span');
      span.style.marginRight = '20px';
      span.textContent = `${predArr[i]}`;
      span.style.color = predArr[i] === labArr[i] ? 'green' : 'red';
      els.previewLabels.appendChild(span);
      xsSplit[i].dispose();
    }
    preds.dispose(); labels.dispose(); batchXs.dispose(); batchYs.dispose();
  } catch (e) {
    console.error(e);
    alert('Preview error: ' + e.message);
  }
}

// ---- Load model from files ----
async function onLoadFromFiles() {
  try {
    const jsonInput = document.getElementById('modelJson');
    const binInput = document.getElementById('modelBin');
    if (!jsonInput.files[0] || !binInput.files[0]) {
      alert('Select model.json and weights.bin');
      return;
    }
    if (state.model) state.model.dispose();
    state.model = await tf.loadLayersModel(tf.io.browserFiles([jsonInput.files[0], binInput.files[0]]));
    els.modelInfo.textContent = 'Loaded model';
  } catch (e) {
    console.error(e);
    alert('Model load error: ' + e.message);
  }
}

// ---- Reset ----
async function onReset(opts = {}) {
  try {
    els.previewRow.innerHTML = '';
    els.previewLabels.innerHTML = '';
    els.metricsText.textContent = '';
    els.modelInfo.textContent = '';
    if (state.model) { state.model.dispose(); state.model = null; }
    for (const key of ['train', 'testClean', 'testNoisy']) {
      if (state[key]?.xs) state[key].xs.dispose();
      if (state[key]?.ys) state[key].ys.dispose();
      state[key] = null;
    }
    if (!opts.keepFiles) {
      state.trainFile = null; state.testFile = null;
      els.inputTrain.value = ''; els.inputTest.value = '';
    }
  } catch (e) {
    console.error(e);
  }
}
