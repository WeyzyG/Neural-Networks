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

    // fallback: WebGL → CPU если WebGL недоступен
    tf.ready().then(async () => {
      try {
        await tf.setBackend('webgl');
      } catch (err) {
        await tf.setBackend('cpu');
      }
    });
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

  async renderEDAforTrain() {
    if (!this.trainData) return;
    const edaDiv = document.getElementById('edaContainer');
    edaDiv.innerHTML = "<h2>EDA analysis of training dataset</h2>";

    // 1. Class distribution chart
    const labels = this.trainData.labels;
    const counts = Array(10).fill(0);
    labels.forEach(lbl => counts[lbl]++);
    edaDiv.innerHTML += '<canvas id="edaClassDistributionChart" width="600" height="300"></canvas><br>';

    // 2. Pixel statistics
    let xsFlatRaw = this.trainData.xs.dataSync ? this.trainData.xs.dataSync() : Array.from(this.trainData.xs);
    let min = Infinity, max = -Infinity, sum = 0, sum2 = 0, validCount = 0;
    for (let v of xsFlatRaw) {
      if (typeof v === 'number' && !isNaN(v) && v >= 0 && v <= 1) {
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        sum2 += v * v;
        validCount += 1;
      }
    }
    const pixelStatsBlock = document.createElement('div');
    pixelStatsBlock.style.margin = '22px 0 18px 0';
    pixelStatsBlock.style.padding = '18px 14px 16px 14px';
    pixelStatsBlock.style.background = '#f7f9fe';
    pixelStatsBlock.style.border = '1.5px solid #e5e8ef';
    pixelStatsBlock.style.borderRadius = '12px';
    pixelStatsBlock.style.boxShadow = '0 2px 8px rgba(65,80,200,0.07)';
    pixelStatsBlock.style.fontSize = '1.08rem';

    if (validCount === 0) {
      pixelStatsBlock.innerHTML = `<b>Pixel Statistics</b><br>No valid pixel data available for analysis.`;
    } else {
      let mean = sum / validCount;
      let std = Math.sqrt(sum2 / validCount - mean * mean);
      pixelStatsBlock.innerHTML = `
        <span style="font-weight:700; font-size:1.17em; color:#4064eb;">Pixel Statistics</span><br>
        <ul style="margin-top:7px; margin-bottom:7px; padding-left:20px;">
          <li>
            <b>Pixel value range:</b> <span style="color:#374ae0;">${min.toFixed(4)} - ${max.toFixed(4)}</span><br>
            Each image contains pixels from <b>0</b> (black, background) to <b>1</b> (white, highlights). Darker pixels dominate, making contours of clothes stand out.
          </li>
          <li>
            <b>Average pixel brightness:</b>
            <span style="color:#4064eb; font-weight:600;">${mean.toFixed(4)}</span><br>
            This means most areas are dim, ensuring your model learns to focus on visible product shapes, not background noise.
          </li>
          <li>
            <b>Variation (std):</b>
            <span style="color:#4064eb; font-weight:600;">${std.toFixed(4)}</span><br>
            High variation in brightness helps the model recognize different clothing types, textures, and materials.
          </li>
        </ul>
        <div style="margin-top:8px; background:#eaf1fd; border-radius:6px; padding:8px 16px; font-weight:500;">
          <b>Fashion MNIST pixel stats show:</b> images have strong contrasts, minimal noise, and diversity, ideal for high-quality neural network training.
        </div>
      `;
    }
    edaDiv.appendChild(pixelStatsBlock);

    // 3. Missing or corrupted (yellow notification & count)
    const nanLabelCount = labels.reduce((acc, l) => acc + (Number.isNaN(l) ? 1 : 0), 0);
    const nanPixelCount = xsFlatRaw.reduce((acc, x) => acc + (typeof x !== "number" || isNaN(x) ? 1 : 0), 0);
    const hasNaN = nanLabelCount > 0 || nanPixelCount > 0;
    const missingBlock = document.createElement('div');
    missingBlock.style.margin = '18px 0 10px 0';
    missingBlock.style.padding = '10px 14px';
    missingBlock.style.background = '#fff7c2';
    missingBlock.style.border = '1.5px solid #e5e8ef';
    missingBlock.style.borderRadius = '10px';
    missingBlock.style.fontSize = '1.05rem';
    missingBlock.style.fontWeight = '500';

    if (hasNaN) {
      missingBlock.innerHTML =
        `<span style="color:#a06503;"><b>Missing or corrupted records detected!</b></span><br>
         <span>Labels with missing values: <b>${nanLabelCount}</b><br>
         Pixels with non-numeric or missing values: <b>${nanPixelCount}</b>
         <br>This may affect model training or evaluation. Please check your data files.</span>`;
    } else {
      missingBlock.innerHTML =
        `<span style="color:#856700;">
          <b>No missing or corrupted records detected.</b>
          <br>All image data and labels are clean and ready for analysis.
        </span>`;
    }
    edaDiv.appendChild(missingBlock);

    // --- 4. Correlation of average pixel intensity between classes ---
    const meansPerClass = [];
    for (let c = 0; c < 10; ++c) {
      const idxs = labels.map((l, i) => l === c ? i : -1).filter(i => i >= 0);
      if (idxs.length > 0) {
        let classSum = 0;
        for (let idx of idxs) {
          let pixels = Array.from(this.trainData.xs.slice([idx,0,0,0],[1,28,28,1]).dataSync());
          let pixelsFiltered = pixels.filter(v => typeof v === 'number' && !isNaN(v) && v >= 0 && v <= 1);
          let pixelsMean = (pixelsFiltered.length > 0)
            ? pixelsFiltered.reduce((a, b) => a + b, 0) / pixelsFiltered.length
            : 0;
          classSum += pixelsMean;
        }
        meansPerClass.push(classSum / idxs.length);
      } else {
        meansPerClass.push(0);
      }
    }
    // Визуализируем средние значения по классам бар-чартом
    const corrChartId = 'edaClassCorrelationChart';
    const corrBlock = document.createElement('div');
    corrBlock.style.margin = '22px 0 17px 0';
    corrBlock.style.padding = '16px 14px';
    corrBlock.style.background = '#faf6fa';
    corrBlock.style.border = '1.5px solid #ede1ee';
    corrBlock.style.borderRadius = '12px';
    corrBlock.style.boxShadow = '0 2px 8px rgba(80,50,160,0.06)';
    corrBlock.innerHTML = `
      <span style="font-weight:700; font-size:1.09em; color:#8328c7;">Class-wise Average Brightness</span><br>
      <canvas id="${corrChartId}" width="600" height="200" style="margin:12px 0 10px 0;"></canvas>
      <div style="margin-bottom:4px;">
        <b>What does it mean?</b>
        <br>
        Each class (type of clothing) has its own average brightness. Classes with similar brightness are harder to distinguish for the model (e.g., T-shirts and shirts), while those with a big difference are easier (e.g., Sandals vs. Ankle boots).
      </div>
      <div style="background:#eddcfd;padding:7px 12px;border-radius:6px;">
        <b>Tip:</b> The more diversity between classes, the more robust your model's feature extraction!
      </div>
    `;
    edaDiv.appendChild(corrBlock);

    setTimeout(() => {
      const ctx1 = document.getElementById('edaClassDistributionChart').getContext('2d');
      new Chart(ctx1, {
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
      const ctx2 = document.getElementById(corrChartId).getContext('2d');
      new Chart(ctx2, {
        type: 'bar',
        data: {
          labels: FASHION_LABELS,
          datasets: [{
            label: 'Average pixel intensity per class',
            data: meansPerClass,
            backgroundColor: '#a062e0'
          }]
        },
        options: {
          responsive: false,
          plugins: {legend: {display: false}},
          scales: { y: {min: 0, max: 1, title: {display: true, text: 'Pixel Intensity (0-1)'}} }
        }
      });
    }, 0);

    // --- 5. Data description ---
    const xsShape = this.trainData.xs.shape;
    const ysShape = this.trainData.ys.shape;
    const dataSummaryBlock = document.createElement('div');
    dataSummaryBlock.style.margin = '22px 0 0 0';
    dataSummaryBlock.style.padding = '15px 14px 12px 14px';
    dataSummaryBlock.style.background = '#f4f9ff';
    dataSummaryBlock.style.border = '1.2px solid #cee3f9';
    dataSummaryBlock.style.borderRadius = '12px';
    dataSummaryBlock.style.boxShadow = '0 1px 6px rgba(30,110,200,0.06)';
    dataSummaryBlock.innerHTML = `
      <span style="font-weight:700; color:#2388d2; font-size:1.1em;">Data Summary</span>
      <ul style="margin-top:6px; margin-bottom:4px; padding-left:18px;">
        <li><b>Number of training samples:</b> ${xsShape[0]}</li>
        <li><b>Image size:</b> ${xsShape[1]} × ${xsShape[2]} pixels</li>
        <li><b>Channels:</b> ${xsShape[3] ? xsShape[3] : 1} (${xsShape[3] === 1 ? "grayscale" : xsShape[3]+" channels"})</li>
        <li><b>Label encoding (one-hot):</b> [${ysShape.join(', ')}] → <b>${ysShape[1]}</b> classes</li>
      </ul>
      <div style="background:#e1f4ff; padding:6px 12px; border-radius:4px;">
          This dataset structure allows neural networks to distinguish fashion item types and train efficiently on clean, standardized image data.
      </div>
    `;
    edaDiv.appendChild(dataSummaryBlock);
  }

  async onLoadData() {
    try {
      const trainFile = document.getElementById('trainFile').files[0];
      const testFile = document.getElementById('testFile').files[0];
      if (!trainFile || !testFile) {
        this.showError('Please select both train and test CSV files');
        return;
      }
      this.showStatus('Loading training data...');
      const trainData = await this.dataLoader.loadTrainFromFiles(trainFile);
      this.showStatus('Loading test data...');
      const testData = await this.dataLoader.loadTestFromFiles(testFile);
      this.trainData = trainData;
      this.testData = testData;
      this.updateDataStatus(trainData.count, testData.count);
      this.showStatus('Data loaded successfully.');
      this.showStatus(`Train shape: ${trainData.xs.shape}, Test shape: ${testData.xs.shape}`);
      this.showStatus(`Train pixel range: min ${trainData.xs.min().dataSync()[0].toFixed(3)}, max ${trainData.xs.max().dataSync()[0].toFixed(3)}`);
      await this.renderEDAforTrain();
    } catch (error) {
      this.showError(`Load error: ${error.message}`);
    }
  }

  async onTrain() {
    if (!this.trainData) {
      this.showError('Please load train data first.');
      return;
    }
    if (this.isTraining) {
      this.showError('Training is already running.');
      return;
    }
    try {
      this.isTraining = true;
      this.showStatus('Building the model...');
      this.model = this.createCNNClassifier();
      this.updateModelInfo();
      this.showStatus('Starting training...');
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
      this.showStatus(`Training completed in ${duration.toFixed(1)}s, final loss: ${finalLoss.toFixed(6)}, val acc: ${finalAcc.toFixed(3)}`);
    } catch (error) {
      this.showError(`Training failed: ${error.message}`);
    } finally {
      this.isTraining = false;
    }
  }

  async onEvaluate() {
    if (!this.model) {
      this.showError('Train or load a model first.');
      return;
    }
    if (!this.testData) {
      this.showError('Load test data first.');
      return;
    }
    try {
      this.showStatus('Evaluating model on test set...');
      const evalOutput = await this.model.evaluate(this.testData.xs, this.testData.ys);
      const testLoss = evalOutput[0].dataSync()[0];
      const testAcc = evalOutput[1].dataSync()[0];
      this.showStatus(`Test set - Loss: ${testLoss.toFixed(4)}, Accuracy: ${(testAcc * 100).toFixed(2)}%`);
    } catch (error) {
      this.showError(`Evaluation failed: ${error.message}`);
    }
  }

  async onTestFive() {
    if (!this.model || !this.testData) {
      this.showError('Please load a model and test data first.');
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
      await this.renderPredictionsPreview(batchXs, trueArray, predArray);
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
      this.showError('No model available for saving.');
      return;
    }
    try {
      await this.model.save('downloads://fashion-mnist-cnn');
      this.showStatus('Model saved successfully.');
    } catch (error) {
      this.showError(`Save failed: ${error.message}`);
    }
  }

  async onLoadFromFiles() {
    const jsonFile = document.getElementById('modelJsonFile').files[0];
    const weightsFile = document.getElementById('modelWeightsFile').files[0];
    if (!jsonFile || !weightsFile) {
      this.showError('Select both model.json and weights.bin files.');
      return;
    }
    try {
      this.showStatus('Loading model from files...');
      if (this.model) this.model.dispose();
      this.model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
      this.updateModelInfo();
      this.showStatus('Model loaded successfully.');
    } catch (error) {
      this.showError(`Model loading failed: ${error.message}`);
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
    this.showStatus('Reset completed.');
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

  async renderPredictionsPreview(images, trueLabels, predLabels) {
    const container = document.getElementById('previewContainer');
    container.innerHTML = '';
    const count = images.shape[0];
    for (let i = 0; i < count; i++) {
      const div = document.createElement('div');
      const canvas = await this.createCanvasFromTensor(images.slice([i, 0, 0, 0], [1, 28, 28, 1]));
      div.appendChild(canvas);
      const info = document.createElement('span');
      info.textContent = `True: ${FASHION_LABELS[trueLabels[i]]} | Predicted: ${FASHION_LABELS[predLabels[i]]}`;
      info.style.marginLeft = '12px';
      div.appendChild(info);
      container.appendChild(div);
    }
  }

  async createCanvasFromTensor(tensor) {
    let imgTensor = tensor.squeeze();
    if (imgTensor.shape.length !== 2) {
      imgTensor = imgTensor.reshape([imgTensor.shape[0], imgTensor.shape[1]]);
    }
    imgTensor = imgTensor.mul(255).cast('int32');
    const [height, width] = imgTensor.shape;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    try {
      await tf.browser.toPixels(imgTensor, canvas);
    } catch (err) {
      canvas.style.background = "#fff8e2";
      const ctx = canvas.getContext("2d");
      ctx.font = "13px monospace";
      ctx.fillStyle = "#b04141";
      ctx.fillText("WebGL Error!", 6, 18);
    }
    return canvas;
  }

  updateModelInfo() {
    const infoEl = document.getElementById('modelInfo');
    if (!this.model) {
      infoEl.innerHTML = '<h3>Model Info</h3><p>No model loaded.</p>';
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
      <p>Number of layers: ${this.model.layers.length}</p>
      <p>Total parameters: ${totalParams.toLocaleString()}</p>
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
