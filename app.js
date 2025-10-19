const FASHION_LABELS = [
  'T-shirt/top',    // 0
  'Trouser',        // 1
  'Pullover',       // 2
  'Dress',          // 3
  'Coat',           // 4
  'Sandal',         // 5
  'Shirt',          // 6
  'Sneaker',        // 7
  'Bag',            // 8
  'Ankle boot'      // 9
];

class MNISTApp {
  constructor() {
    this.dataLoader = new MNISTDataLoader();
    this.model = null;
    this.isTraining = false;
    this.trainData = null;
    this.testData = null;

    this.initializeUI();
  }

  initializeUI() {
    document.getElementById('loadDataBtn').addEventListener('click', () => this.onLoadData());
    document.getElementById('trainBtn').addEventListener('click', () => this.onTrain());
    document.getElementById('evaluateBtn').addEventListener('click', () => this.onEvaluate());
    document.getElementById('testFiveBtn').addEventListener('click', () => this.onTestFive());
    document.getElementById('saveModelBtn').addEventListener('click', () => this.onSaveDownload());
    document.getElementById('loadModelBtn').addEventListener('click', () => this.onLoadFromFiles());
    document.getElementById('resetBtn').addEventListener('click', () => this.onReset());
    document.getElementById('toggleVisorBtn').addEventListener('click', () => this.toggleVisor());
  }

  async onLoadData() {
    try {
      const trainFile = document.getElementById('trainFile').files[0];
      const testFile = document.getElementById('testFile').files[0];
      if (!trainFile || !testFile) {
        this.showError('Select train and test CSV files');
        return;
      }
      this.showStatus('Loading training data...');
      const trainData = await this.dataLoader.loadTrainFromFiles(trainFile);
      this.showStatus('Loading test data...');
      const testData = await this.dataLoader.loadTestFromFiles(testFile);

      this.trainData = trainData;
      this.testData = testData;

      this.updateDataStatus(trainData.count, testData.count);
      this.showStatus('Data loaded');
      this.showStatus(`Train shape: ${trainData.xs.shape}, Test shape: ${testData.xs.shape}`);
      this.showStatus(`Data range - Min: ${trainData.xs.min().dataSync()[0].toFixed(3)}, Max: ${trainData.xs.max().dataSync()[0].toFixed(3)}`);

      // Автоматически запускаем EDA
      this.runEDA();

    } catch (error) {
      this.showError(`Load error: ${error.message}`);
    }
  }

  async onTrain() {
    if (!this.trainData) {
      this.showError('Load train data first');
      return;
    }
    if (this.isTraining) {
      this.showError('Training already started');
      return;
    }
    try {
      this.isTraining = true;
      this.showStatus('Building Fashion-MNIST classifier...');
      this.model = this.createCNNClassifier();
      this.updateModelInfo();

      this.showStatus('Training...');
      const startTime = Date.now();
      const history = await this.model.fit(this.trainData.xs, this.trainData.ys, {
        epochs: 5,
        batchSize: 32,
        validationSplit: 0.2,
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            this.showStatus(`Epoch ${epoch + 1}/5 - loss: ${logs.loss.toFixed(6)}, val_acc: ${(logs.val_acc || logs.val_accuracy).toFixed(3)}`);
          }
        }
      });

      const duration = (Date.now() - startTime) / 1000;
      const finalLoss = history.history.loss[history.history.loss.length - 1];
      const finalAcc = (history.history.val_acc || history.history.val_accuracy)[history.history.val_loss.length - 1];
      this.showStatus(`Training done in ${duration.toFixed(1)}s, final loss: ${finalLoss.toFixed(6)}, val acc: ${finalAcc.toFixed(3)}`);
    } catch (error) {
      this.showError(`Train failed: ${error.message}`);
    } finally {
      this.isTraining = false;
    }
  }

  async onEvaluate() {
    if (!this.model) {
      this.showError('Train or load model first');
      return;
    }
    if (!this.testData) {
      this.showError('Load test data first');
      return;
    }
    try {
      this.showStatus('Evaluating on test set...');
      const evalOutput = await this.model.evaluate(this.testData.xs, this.testData.ys);
      const testLoss = evalOutput[0].dataSync()[0];
      const testAcc = evalOutput[1].dataSync()[0];
      this.showStatus(`Test Loss: ${testLoss.toFixed(4)}, Test Accuracy: ${(testAcc * 100).toFixed(2)}%`);
    } catch (error) {
      this.showError(`Eval failed: ${error.message}`);
    }
  }

  async onTestFive() {
    if (!this.model || !this.testData) {
      this.showError('Train or load model and test data first');
      return;
    }
    try {
      this.showStatus('Predicting 5 random test images...');
      const { batchXs, batchYs, indices } = this.dataLoader.getRandomTestBatch(this.testData.xs, this.testData.ys, 5);

      const preds = this.model.predict(batchXs);
      const predLabels = preds.argMax(-1);
      const trueLabels = batchYs.argMax(-1);
      const predArray = await predLabels.array();
      const trueArray = await trueLabels.array();

      this.renderPredictionsPreview(batchXs, trueArray, predArray);

      batchXs.dispose();
      batchYs.dispose();
      preds.dispose();
      predLabels.dispose();
      trueLabels.dispose();
    } catch (error) {
      this.showError(`Test preview failed: ${error.message}`);
    }
  }

  async onSaveDownload() {
    if (!this.model) {
      this.showError('No model to save');
      return;
    }
    try {
      await this.model.save('downloads://fashion-mnist-cnn');
      this.showStatus('Model saved');
    } catch (error) {
      this.showError(`Save fail: ${error.message}`);
    }
  }

  async onLoadFromFiles() {
    const jsonFile = document.getElementById('modelJsonFile').files[0];
    const weightsFile = document.getElementById('modelWeightsFile').files[0];
    if (!jsonFile || !weightsFile) {
      this.showError('Select model.json and weights.bin');
      return;
    }
    try {
      this.showStatus('Loading model...');
      if (this.model) this.model.dispose();
      this.model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
      this.updateModelInfo();
      this.showStatus('Model loaded');
    } catch (error) {
      this.showError(`Load fail: ${error.message}`);
    }
  }

  onReset() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.dataLoader.dispose();
    this.trainData = null;
    this.testData = null;
    this.updateDataStatus(0, 0);
    this.updateModelInfo();
    this.clearPreview();
    this.showStatus('Reset done');
    document.getElementById('edaStats').innerHTML = '';
    document.getElementById('edaClassDistribution').innerHTML = '';
    document.getElementById('edaImagesGrid').innerHTML = '';
  }

  toggleVisor() {
    tfvis.visor().toggle();
  }

  createCNNClassifier() {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.dropout({ rate: 0.25 }));

    model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.dropout({ rate: 0.25 }));

    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));

    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    model.summary();
    return model;
  }

  // ---- EDA SECTION ----
  runEDA() {
    const statsDiv = document.getElementById('edaStats');
    const classDistDiv = document.getElementById('edaClassDistribution');
    const imagesDiv = document.getElementById('edaImagesGrid');
    statsDiv.innerHTML = '<b>Data shapes:</b><br>';

    // 6. Shape данных
    const trainShape = this.trainData.xs.shape;
    const testShape = this.testData.xs.shape;
    statsDiv.innerHTML += `Train: ${trainShape}, Test: ${testShape}<br>`;

    // 2. Статистика пикселей (Train/Test)
    Promise.all([
      this.trainData.xs.min().data(),
      this.trainData.xs.max().data(),
      this.trainData.xs.mean().data(),
      this.trainData.xs.flatten().array(),
      this.testData.xs.min().data(),
      this.testData.xs.max().data(),
      this.testData.xs.mean().data(),
      this.testData.xs.flatten().array(),
    ]).then(([tMin,tMax,tMean,tArray,teMin,teMax,teMean,teArray]) => {
      const trainStd = this.std(tArray, tMean[0]);
      const testStd = this.std(teArray, teMean[0]);
      statsDiv.innerHTML += `<b>Train pixels:</b> min: ${tMin[0].toFixed(3)} max: ${tMax[0].toFixed(3)} mean: ${tMean[0].toFixed(3)} std: ${trainStd.toFixed(3)}<br>`;
      statsDiv.innerHTML += `<b>Test pixels:</b> min: ${teMin[0].toFixed(3)} max: ${teMax[0].toFixed(3)} mean: ${teMean[0].toFixed(3)} std: ${testStd.toFixed(3)}<br>`;

      // 4. NaN значения
      const trainHasNaN = tArray.some(x => isNaN(x));
      const testHasNaN = teArray.some(x => isNaN(x));
      statsDiv.innerHTML += `<b>Missing values (NaN):</b> Train ${trainHasNaN ? 'YES' : 'NO'}, Test ${testHasNaN ? 'YES' : 'NO'}<br>`;
    });

    // 1. Распределение классов (Train/Test)
    classDistDiv.innerHTML = '<b>Class Distribution:</b><br>';
    const trDist = this.getClassDist(this.trainData.labels);
    const teDist = this.getClassDist(this.testData.labels);
    for (let i = 0; i < FASHION_LABELS.length; i++) {
      classDistDiv.innerHTML += FASHION_LABELS[i] + ` (train: ${trDist[i]}, test: ${teDist[i]})<br>`;
    }

    // 3. Примеры изображений по классам (Train set, по одному на класс)
    imagesDiv.innerHTML = '<b>Class Examples (train):</b><br>';
    for (let label = 0; label < 10; label++) {
      const idx = this.trainData.labels.findIndex(l => l === label);
      if (idx >= 0) {
        const imgTensor = this.trainData.xs.slice([idx,0,0,0],[1,28,28,1]);
        const canvas = document.createElement('canvas');
        canvas.width = 28; canvas.height = 28;
        tf.browser.toPixels(imgTensor.squeeze(), canvas);
        imagesDiv.appendChild(canvas);
        imagesDiv.innerHTML += ` ${FASHION_LABELS[label]} `;
        imgTensor.dispose();
      }
    }

    // 5. Корреляция между классами (по средней интенсивности классов, только train)
    const corrArr = [];
    for (let i = 0; i < 10; i++) {
      const idxs = this.trainData.labels.reduce((arr, l, k) => (l === i ? arr.concat([k]) : arr), []);
      if (idxs.length > 1) {
        const tensors = tf.gather(this.trainData.xs, idxs);
        const meanImg = tensors.mean(0).flatten();
        corrArr.push(meanImg.arraySync());
        tensors.dispose();
      }
    }
    let maxCorr = -1, minCorr = 2, pair = '';
    for (let i = 0; i < 10; i++) for (let j = i+1; j < 10; j++) {
      const c = this.pearsonCorr(corrArr[i], corrArr[j]);
      if (c > maxCorr) { maxCorr = c; pair = `${FASHION_LABELS[i]} / ${FASHION_LABELS[j]}`;}
      if (c < minCorr) { minCorr = c; }
    }
    statsDiv.innerHTML += `<br><b>Max correlation between class averages:</b> ${pair} = ${maxCorr.toFixed(3)}<br>`;
  }

  getClassDist(labels) {
    const dist = Array(10).fill(0);
    labels.forEach(l => dist[l]++);
    return dist;
  }

  std(arr, mean) {
    return Math.sqrt(arr.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / arr.length);
  }

  pearsonCorr(a, b) {
    const n = a.length;
    const ma = a.reduce((ac, v) => ac + v, 0) / n;
    const mb = b.reduce((ac, v) => ac + v, 0) / n;
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < n; i++) {
      num += (a[i]-ma)*(b[i]-mb);
      da += (a[i]-ma)**2;
      db += (b[i]-mb)**2;
    }
    return num / Math.sqrt(da*db);
  }

  renderPredictionsPreview(images, trueLabels, predLabels) {
    const container = document.getElementById('previewContainer');
    container.innerHTML = '';
    const count = images.shape[0];
    for (let i = 0; i < count; i++) {
      const div = document.createElement('div');
      const canvas = this.createCanvasFromTensor(images.slice([i, 0, 0, 0], [1, 28, 28, 1]));
      div.appendChild(canvas);
      const info = document.createElement('span');
      info.textContent = `True: ${FASHION_LABELS[trueLabels[i]]} | Pred: ${FASHION_LABELS[predLabels[i]]}`;
      info.style.marginLeft = '12px';
      div.appendChild(info);
      container.appendChild(div);
    }
  }

  createCanvasFromTensor(tensor) {
    const [height, width] = tensor.shape.slice(1, 3);
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    tf.browser.toPixels(tensor.squeeze(), canvas);
    return canvas;
  }

  updateModelInfo() {
    const infoEl = document.getElementById('modelInfo');
    if (!this.model) {
      infoEl.innerHTML = '<h3>Model Info</h3><p>No model loaded</p>';
      return;
    }
    let totalParams = 0;
    this.model.layers.forEach(layer => {
      layer.getWeights().forEach(weight => {
        totalParams += weight.size;
      });
    });
    infoEl.innerHTML = `
      <h3>Model Info</h3>
      <p>Layers: ${this.model.layers.length}</p>
      <p>Params: ${totalParams.toLocaleString()}</p>
    `;
  }

  updateDataStatus(trainCount, testCount) {
    const el = document.getElementById('dataStatus');
    el.innerHTML = `<h3>Data Status</h3><p>Training samples: ${trainCount}</p><p>Test samples: ${testCount}</p>`;
  }

  showStatus(message) {
    const logs = document.getElementById('trainingLogs');
    const entry = document.createElement('div');
    entry.textContent = `[info] ${message}`;
    logs.appendChild(entry);
    logs.scrollTop = logs.scrollHeight;
  }

  showError(message) {
    const logs = document.getElementById('trainingLogs');
    const entry = document.createElement('div');
    entry.style.color = 'red';
    entry.textContent = `[error] ${message}`;
    logs.appendChild(entry);
    logs.scrollTop = logs.scrollHeight;
    console.error(message);
  }

  clearPreview() {
    const container = document.getElementById('previewContainer');
    container.innerHTML = '';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new MNISTApp();
});
