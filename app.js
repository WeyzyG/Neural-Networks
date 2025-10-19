const FASHION_LABELS = [
  'T-shirt/top', // 0
  'Trouser',     // 1
  'Pullover',    // 2
  'Dress',       // 3
  'Coat',        // 4
  'Sandal',      // 5
  'Shirt',       // 6
  'Sneaker',     // 7
  'Bag',         // 8
  'Ankle boot'   // 9
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

  // ------- EDA ВСТАВКА ПРОСТОЙ --------
  async renderEDAforTrain() {
    if (!this.trainData) return;
    const edaDiv = document.getElementById('edaContainer');
    edaDiv.innerHTML = "<h2>EDA анализ тренировочного датасета</h2>";

    // 1. Гистограмма распределения классов
    const labels = this.trainData.labels;
    const counts = Array(10).fill(0);
    labels.forEach(lbl => counts[lbl]++);
    let hist = '<b>Распределение классов (label count):</b><br>';
    hist += '<canvas id="edaClassDistributionChart" width="340" height="120"></canvas><br>';
    edaDiv.innerHTML += hist;

    // 2. Статистика по интенсивности
    const xsFlat = this.trainData.xs.dataSync();
    let min = xsFlat[0], max = xsFlat[0], sum = 0, sum2 = 0;
    for (let v of xsFlat) {
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      sum2 += v*v;
    }
    const mean = sum / xsFlat.length;
    const std = Math.sqrt(sum2 / xsFlat.length - mean * mean);
    edaDiv.innerHTML += `<b>Пиксельная статистика:</b><br>
      min: ${min.toFixed(4)}, max: ${max.toFixed(4)}, mean: ${mean.toFixed(4)}, std: ${std.toFixed(4)}<br>`;

    // 3. Примеры изображений из каждого класса
    edaDiv.innerHTML += `<b>Примеры изображений по классам:</b>
      <div id="edaImagesGrid"></div>`;
    const grid = document.getElementById('edaImagesGrid');
    for (let classIdx = 0; classIdx < 10; ++classIdx) {
      // ищем индексы примеров текущего класса
      const indices = labels
        .map((lbl, i) => lbl === classIdx ? i : -1)
        .filter(i => i >= 0);
      if (indices.length === 0) continue;
      for (let i = 0; i < Math.min(2, indices.length); ++i) {
        const canvas = document.createElement('canvas');
        canvas.width = 28; canvas.height = 28;
        this.dataLoader.draw28x28ToCanvas(
          this.trainData.xs.slice([indices[i], 0, 0, 0], [1, 28, 28, 1]), 
          canvas, 3
        );
        canvas.title = `Label ${classIdx}`;
        grid.appendChild(canvas);
      }
    }

    // 4. Проверка на NaN или некорректные лейблы
    const hasNaN = labels.some(l => Number.isNaN(l)) || Array.from(xsFlat).some(x => Number.isNaN(x));
    edaDiv.innerHTML += `<br><b>Отсутствующие значения или битые записи:</b> ${hasNaN ? "Обнаружены!" : "Нет"}<br>`;

    // 5. (опционально) Корреляция между классами
    let corrBlock = '';
    const meansPerClass = [];
    for(let c = 0; c < 10; ++c){
      const idxs = labels.map((l,i) => l === c ? i : -1).filter(i => i >= 0);
      if (idxs.length > 0){
        let sum = 0;
        for (let idx of idxs){
          sum += this.trainData.xs.slice([idx,0,0,0],[1,28,28,1]).mean().dataSync()[0];
        }
        meansPerClass.push(sum/idxs.length);
      } else meansPerClass.push(0);
    }
    const corr = meansPerClass.map((v,i,arr) => ((i<arr.length-1)? (Math.abs(v-arr[i+1])).toFixed(4) : "-")).join(', ');
    corrBlock += `<b>Корреляция средних интенсивностей по соседним классам:</b> ${corr}<br>`;
    edaDiv.innerHTML += corrBlock;

    // 6. Shape данных
    edaDiv.innerHTML += `<b>Shape данных:</b> xs: ${this.trainData.xs.shape.join(', ')}, ys: ${this.trainData.ys.shape.join(', ')} <br>`;

    // отрисовка гистограммы
    setTimeout(() => {
      const ctx = document.getElementById('edaClassDistributionChart').getContext('2d');
      new Chart(ctx, {
        type: 'bar',
        data: {
          labels: FASHION_LABELS,
          datasets: [{
            label: 'Samples per class',
            data: counts,
            backgroundColor: '#007BFF',
          }]
        },
        options: {responsive:false, plugins:{legend:{display:false}}}
      });
    }, 0);
  }
  // -------------------------------------

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
      await this.renderEDAforTrain(); // <--- ВЫЗОВ EDA после загрузки
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
      const history = await this.model.fit(
        this.trainData.xs, 
        this.trainData.ys, 
        {
          epochs: 5,
          batchSize: 32,
          validationSplit: 0.2,
          shuffle: true,
          callbacks: {
            onEpochEnd: (epoch, logs) => {
              this.showStatus(`Epoch ${epoch + 1}/5 - loss: ${logs.loss.toFixed(6)}, val_acc: ${(logs.val_acc || logs.val_accuracy).toFixed(3)}`);
            }
          }
        }
      );
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
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    model.summary();
    return model;
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
    canvas.width = width; canvas.height = height;
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
    el.innerHTML = `<h3>Data Status</h3>
      <p>Training samples: ${trainCount}</p>
      <p>Test samples: ${testCount}</p>`;
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
