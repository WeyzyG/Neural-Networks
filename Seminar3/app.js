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
      this.showStatus('Creating simple autoencoder...');
      this.model = this.createSimpleWorkingAutoencoder();
      this.updateModelInfo();

      this.showStatus('Testing model...');
      const testInput = this.trainData.xs.slice([0, 0, 0, 0], [1, 28, 28, 1]);
      const testOutput = this.model.predict(testInput);
      this.showStatus(`Model test ok: input shape ${testInput.shape}, output shape ${testOutput.shape}`);

      const outputData = await testOutput.data();
      this.showStatus(`Output range min:${Math.min(...outputData).toFixed(3)}, max:${Math.max(...outputData).toFixed(3)}`);

      testInput.dispose();
      testOutput.dispose();

      this.showStatus('Training started');
      const trainSubset = this.trainData.xs.slice([0, 0, 0, 0], [500, 28, 28, 1]);
      const noisyData = this.dataLoader.addNoise(trainSubset, 0.1);

      const noisyMin = (await noisyData.min().data())[0];
      const noisyMax = (await noisyData.max().data())[0];
      this.showStatus(`Noisy data range: ${noisyMin.toFixed(3)} - ${noisyMax.toFixed(3)}`);

      const startTime = Date.now();
      const history = await this.model.fit(noisyData, trainSubset, {
        epochs: 5,
        batchSize: 16,
        validationSplit: 0.2,
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            this.showStatus(`Epoch ${epoch + 1}/5 - loss: ${logs.loss.toFixed(6)}, val_loss: ${logs.val_loss.toFixed(6)}`);
            if (epoch % 2 === 0) setTimeout(() => this.quickTest(), 100);
          }
        }
      });

      const duration = (Date.now() - startTime) / 1000;
      const finalLoss = history.history.loss[history.history.loss.length - 1];
      this.showStatus(`Training done in ${duration.toFixed(1)}s`);
      this.showStatus(`Final loss: ${finalLoss.toFixed(6)}`);

      if (finalLoss < 0.01) this.showStatus('Training seems ok');
      else this.showStatus('Loss still high, consider more training');

      trainSubset.dispose();
      noisyData.dispose();

      await this.onTestFive();

    } catch (error) {
      this.showError(`Train failed: ${error.message}`);
      console.error(error);
    } finally {
      this.isTraining = false;
    }
  }

  async quickTest() {
    if (!this.model || !this.testData) return;
    try {
      const { batchXs, batchYs, indices } = this.dataLoader.getRandomTestBatch(this.testData.xs, this.testData.ys, 2);
      const noisyBatchXs = this.dataLoader.addNoise(batchXs, 0.1);
      const reconstructions = this.model.predict(noisyBatchXs);

      const reconData = await reconstructions.data();
      const avgOutput = reconData.reduce((a, b) => a + b) / reconData.length;
      this.showStatus(`Quick test avg output: ${avgOutput.toFixed(3)}`);

      noisyBatchXs.dispose();
      reconstructions.dispose();
      batchXs.dispose();
      batchYs.dispose();

    } catch (error) {
      this.showError(`Quick test failed: ${error.message}`);
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
      this.showStatus('Starting evaluation...');
      const testSubset = this.testData.xs.slice([0, 0, 0, 0], [100, 28, 28, 1]);
      const noisyData = this.dataLoader.addNoise(testSubset, 0.1);
      const reconstructions = this.model.predict(noisyData);

      const originalData = await testSubset.data();
      const reconData = await reconstructions.data();

      const mse = reconstructions.sub(testSubset).square().mean();
      const mseValue = (await mse.data())[0];

      const avgOriginal = originalData.reduce((a, b) => a + b) / originalData.length;
      const avgRecon = reconData.reduce((a, b) => a + b) / reconData.length;

      this.showStatus('Evaluation results:');
      this.showStatus(`Original avg: ${avgOriginal.toFixed(3)}, Reconstructed avg: ${avgRecon.toFixed(3)}`);
      this.showStatus(`MSE: ${mseValue.toFixed(6)}`);

      if (mseValue < 0.01) this.showStatus('Model OK');
      else if (mseValue < 0.05) this.showStatus('Model needs more train');
      else this.showStatus('Model not learned');

      testSubset.dispose();
      noisyData.dispose();
      reconstructions.dispose();
      mse.dispose();

    } catch (error) {
      this.showError(`Eval failed: ${error.message}`);
    }
  }

  async onTestFive() {
    if (!this.model || !this.testData) {
      this.showError('Load model and test data first');
      return;
    }
    try {
      this.showStatus('Running denoise test...');
      const { batchXs, batchYs, indices } = this.dataLoader.getRandomTestBatch(this.testData.xs, this.testData.ys, 5);
      const noisyBatchXs = this.dataLoader.addNoise(batchXs, 0.1);
      const reconstructions = this.model.predict(noisyBatchXs);

      const reconData = await reconstructions.data();
      const hasValidOutput = reconData.some(val => val > 0.1 && val < 0.9);
      if (!hasValidOutput) this.showStatus('Warning: Model output suspicious');

      const trueLabels = batchYs.argMax(-1);
      const trueArray = await trueLabels.array();

      this.renderDenoisingPreview(batchXs, noisyBatchXs, reconstructions, trueArray, indices);

      noisyBatchXs.dispose();
      reconstructions.dispose();
      batchXs.dispose();
      batchYs.dispose();
      trueLabels.dispose();

    } catch (error) {
      this.showError(`Denoise test fail: ${error.message}`);
    }
  }

  async onSaveDownload() {
    if (!this.model) {
      this.showError('No model to save');
      return;
    }
    try {
      await this.model.save('downloads://mnist-autencoder');
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

  createSimpleWorkingAutoencoder() {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      filters: 4,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
      name: 'conv1'
    }));

    model.add(tf.layers.conv2d({
      filters: 4,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
      name: 'conv2'
    }));

    model.add(tf.layers.conv2d({
      filters: 1,
      kernelSize: 3,
      activation: 'sigmoid',
      padding: 'same',
      name: 'output'
    }));

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mse']
    });

    console.log('Autoencoder built');
    model.summary();

    return model;
  }

  renderDenoisingPreview(original, noisy, reconstructed, trueLabels, indices) {
    const container = document.getElementById('previewContainer');
    container.innerHTML = ''; // Clear previous

    const count = original.shape[0];

    for (let i = 0; i < count; i++) {
      const div = document.createElement('div');

      const origCanvas = this.createCanvasFromTensor(original.slice([i, 0, 0, 0], [1, 28, 28, 1]));
      const noisyCanvas = this.createCanvasFromTensor(noisy.slice([i, 0, 0, 0], [1, 28, 28, 1]));
      const reconCanvas = this.createCanvasFromTensor(reconstructed.slice([i, 0, 0, 0], [1, 28, 28, 1]));

      div.appendChild(origCanvas);
      div.appendChild(noisyCanvas);
      div.appendChild(reconCanvas);

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

// Init app on page load
document.addEventListener('DOMContentLoaded', () => {
  new MNISTApp();
});
