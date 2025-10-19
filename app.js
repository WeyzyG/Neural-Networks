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
        this.showError('Выберите train и test CSV файлы');
        return;
      }

      this.showStatus('Загрузка тренировочных данных...');
      const trainData = await this.dataLoader.loadTrainFromFiles(trainFile);

      this.showStatus('Загрузка тестовых данных...');
      const testData = await this.dataLoader.loadTestFromFiles(testFile);

      this.trainData = trainData;
      this.testData = testData;

      this.updateDataStatus(trainData.count, testData.count);
      this.showStatus('Данные загружены');
      this.showStatus(`Train shape: ${trainData.xs.shape}, Test shape: ${testData.xs.shape}`);
      this.showStatus(`Диапазон данных - Min: ${trainData.xs.min().dataSync()[0].toFixed(3)}, Max: ${trainData.xs.max().dataSync()[0].toFixed(3)}`);
    } catch (error) {
      this.showError(`Ошибка загрузки: ${error.message}`);
    }
  }

  async onTrain() {
    if (!this.trainData) {
      this.showError('Сначала загрузите тренировочные данные');
      return;
    }
    if (this.isTraining) {
      this.showError('Обучение уже запущено');
      return;
    }
    try {
      this.isTraining = true;
      this.showStatus('Создание CNN классификатора...');
      this.model = this.createCNNClassifier();
      this.updateModelInfo();

      this.showStatus('Запуск обучения...');
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
      this.showStatus(`Обучение закончено за ${duration.toFixed(1)}s, финальная loss: ${finalLoss.toFixed(6)}, val acc: ${finalAcc.toFixed(3)}`);

    } catch (error) {
      this.showError(`Ошибка обучения: ${error.message}`);
      console.error(error);
    } finally {
      this.isTraining = false;
    }
  }

  async onEvaluate() {
    if (!this.model) {
      this.showError('Сначала обучите или загрузите модель');
      return;
    }
    if (!this.testData) {
      this.showError('Сначала загрузите тестовые данные');
      return;
    }
    try {
      this.showStatus('Оценка на тесте...');
      const evalOutput = await this.model.evaluate(this.testData.xs, this.testData.ys);
      const testLoss = evalOutput[0].dataSync()[0];
      const testAcc = evalOutput[1].dataSync()[0];
      this.showStatus(`Test Loss: ${testLoss.toFixed(4)}, Test Accuracy: ${(testAcc * 100).toFixed(2)}%`);
    } catch (error) {
      this.showError(`Ошибка оценки: ${error.message}`);
    }
  }

  async onTestFive() {
    if (!this.model || !this.testData) {
      this.showError('Сначала загрузите модель и тестовые данные');
      return;
    }
    try {
      this.showStatus('Вывод примера предсказаний...');
      const { batchXs, batchYs, indices } = this.dataLoader.getRandomTestBatch(this.testData.xs, this.testData.ys, 5);

      const preds = this.model.predict(batchXs);
      const predLabels = preds.argMax(-1);
      const trueLabels = batchYs.argMax(-1);

      const predArray = await predLabels.array();
      const trueArray = await trueLabels.array();

      this.renderPredictionsPreview(batchXs, trueArray, predArray, indices);

      batchXs.dispose();
      batchYs.dispose();
      preds.dispose();
      predLabels.dispose();
      trueLabels.dispose();

    } catch (error) {
      this.showError(`Ошибка предсказаний: ${error.message}`);
    }
  }

  async onSaveDownload() {
    if (!this.model) {
      this.showError('Нет модели для сохранения');
      return;
    }
    try {
      await this.model.save('downloads://fashion-mnist-cnn');
      this.showStatus('Модель успешно сохранена');
    } catch (error) {
      this.showError(`Ошибка сохранения: ${error.message}`);
    }
  }

  async onLoadFromFiles() {
    const jsonFile = document.getElementById('modelJsonFile').files[0];
    const weightsFile = document.getElementById('modelWeightsFile').files[0];

    if (!jsonFile || !weightsFile) {
      this.showError('Выберите model.json и weights.bin');
      return;
    }
    try {
      this.showStatus('Загрузка модели...');
      if (this.model) this.model.dispose();

      this.model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
      this.updateModelInfo();
      this.showStatus('Модель загружена');

    } catch (error) {
      this.showError(`Ошибка загрузки: ${error.message}`);
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
    this.showStatus('Сброс завершён');
  }

  toggleVisor() {
    tfvis.visor().toggle();
  }

  // CNN классификатор для 10 классов
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

  renderPredictionsPreview(images, trueLabels, predLabels, indices) {
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
      infoEl.innerHTML = '<h3>Model Info</h3><p>Нет загруженной модели</p>';
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
