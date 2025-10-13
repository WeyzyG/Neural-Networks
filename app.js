class MNISTApp {
  constructor() {
    this.dataLoader = new MNISTDataLoader();
    this.model = null;
    this.isTraining = false;
    this.trainData = null;
    this.testData = null;
    document.addEventListener('DOMContentLoaded', () => {});
    this.initializeUI();
  }

  // Привязка событий к новой разметке
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

  // ===================== Загрузка данных =====================
  async onLoadData() {
    try {
      const trainFile = document.getElementById('trainFile').files[0];
      const testFile = document.getElementById('testFile').files[0];
      if (!trainFile || !testFile) {
        this.showError('Выберите оба файла: train и test CSV');
        return;
      }
      this.showStatus('Загрузка train данных...');
      const trainData = await this.dataLoader.loadTrainFromFiles(trainFile);
      this.showStatus('Загрузка test данных...');
      const testData = await this.dataLoader.loadTestFromFiles(testFile);
      this.trainData = trainData;
      this.testData = testData;
      this.updateDataStatus(trainData.count, testData.count);
      this.showStatus('✓ Данные загружены');
      this.showStatus(`Train shape: ${trainData.xs.shape}, Test shape: ${testData.xs.shape}`);
    } catch (error) {
      this.showError(`Сбой загрузки данных: ${error.message}`);
    }
  }

  // ===================== Обучение модели =====================
  async onTrain() {
    if (!this.trainData) { this.showError('Сначала загрузите train данные'); return; }
    if (this.isTraining) { this.showError('Обучение уже запущено'); return; }
    try {
      this.isTraining = true;
      this.showStatus('Создание упрощённого автоэнкодера...');
      this.model = this.createSimpleAutoencoder();
      this.updateModelInfo();

      // Быстрая проверка формы выхода
      const testInput = this.trainData.xs.slice([0,0,0,0],[1,28,28,1]);
      const testOutput = this.model.predict(testInput);
      this.showStatus(`Проверка вывода: ${testInput.shape} → ${testOutput.shape}`);
      testInput.dispose(); testOutput.dispose();

      // Небольшой сабсет
      const trainSubset = this.trainData.xs.slice([0,0,0,0],[500,28,28,1]);
      const noisy = this.dataLoader.addNoise(trainSubset, 0.1);
      this.showStatus('Старт обучения (5 эпох)');

      const history = await this.model.fit(noisy, trainSubset, {
        epochs: 5, batchSize: 16, validationSplit: 0.2, shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            this.showStatus(`Эпоха ${epoch+1}/5 - loss: ${logs.loss.toFixed(6)}, val_loss: ${logs.val_loss.toFixed(6)}`);
            if (epoch % 2 === 0) setTimeout(()=>this.quickTest(), 50);
          }
        }
      });

      const finalLoss = history.history.loss.at(-1);
      this.showStatus(`Готово: loss=${finalLoss.toFixed(6)}`);
      trainSubset.dispose(); noisy.dispose();

      await this.onTestFive();
    } catch (error) {
      this.showError(`Ошибка обучения: ${error.message}`);
    } finally {
      this.isTraining = false;
    }
  }

  async quickTest() {
    if (!this.model || !this.testData) return;
    try {
      const { batchXs, batchYs } = this.dataLoader.getRandomTestBatch(this.testData.xs, this.testData.ys, 2);
      const noisy = this.dataLoader.addNoise(batchXs, 0.1);
      const recon = this.model.predict(noisy);
      const avg = (await recon.data()).reduce((a,b)=>a+b,0)/recon.size;
      this.showStatus(`Quick test: avg=${avg.toFixed(3)}`);
      noisy.dispose(); recon.dispose(); batchXs.dispose(); batchYs.dispose();
    } catch(e){ /* no-op */ }
  }

  // ===================== Оценка =====================
  async onEvaluate() {
    if (!this.model) { this.showError('Нет модели. Обучите или загрузите её.'); return; }
    if (!this.testData) { this.showError('Нет test данных'); return; }
    try {
      this.showStatus('Оценка на 100 примерах...');
      const testSubset = this.testData.xs.slice([0,0,0,0],[100,28,28,1]);
      const noisy = this.dataLoader.addNoise(testSubset, 0.1);
      const recon = this.model.predict(noisy);

      const { mse, psnr } = this.dataLoader.calculatePSNR(testSubset, recon);
      const mseVal = (await mse.data())[0];
      const psnrVal = (await psnr.data())[0];
      this.renderMetrics({ mse: mseVal, psnr: psnrVal, n: 100 });

      testSubset.dispose(); noisy.dispose(); recon.dispose(); mse.dispose(); psnr.dispose();
      this.showStatus(`MSE=${mseVal.toFixed(6)}, PSNR=${psnrVal.toFixed(2)} dB`);
    } catch (error) {
      this.showError(`Сбой оценки: ${error.message}`);
    }
  }

  // ===================== Предпросмотр ×5 =====================
  async onTestFive() {
    if (!this.model || !this.testData) { this.showError('Загрузите модель и тестовые данные'); return; }
    try {
      this.showStatus('Предпросмотр денойзинга (5 образцов)...');
      const { batchXs, batchYs, indices } = this.dataLoader.getRandomTestBatch(this.testData.xs, this.testData.ys, 5);
      const noisy = this.dataLoader.addNoise(batchXs, 0.1);
      const recon = this.model.predict(noisy);

      const trueLabels = batchYs.argMax(-1);
      const labels = await trueLabels.array();
      this.renderDenoisingPreview(batchXs, noisy, recon, labels, indices);

      const { mse, psnr } = this.dataLoader.calculatePSNR(batchXs, recon);
      const mseVal = (await mse.data())[0];
      const psnrVal = (await psnr.data())[0];
      this.renderMetrics({ mse: mseVal, psnr: psnrVal, n: labels.length });

      noisy.dispose(); recon.dispose(); batchXs.dispose(); batchYs.dispose(); trueLabels.dispose(); mse.dispose(); psnr.dispose();
    } catch (error) {
      this.showError(`Сбой предпросмотра: ${error.message}`);
    }
  }

  // ===================== Сохранение/загрузка модели =====================
  async onSaveDownload() {
    if (!this.model) { this.showError('Нет модели для сохранения'); return; }
    try {
      await this.model.save('downloads://mnist-denoiser');
      this.showStatus('Модель сохранена');
    } catch (error) {
      this.showError(`Сбой сохранения: ${error.message}`);
    }
  }

  async onLoadFromFiles() {
    const jsonFile = document.getElementById('modelJsonFile').files[0];
    const weightsFile = document.getElementById('modelWeightsFile').files[0];
    if (!jsonFile || !weightsFile) { this.showError('Выберите model.json и weights.bin'); return; }
    try {
      this.showStatus('Загрузка модели из файлов...');
      if (this.model) this.model.dispose();
      this.model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
      this.updateModelInfo();
      this.showStatus('Модель загружена');
    } catch (error) {
      this.showError(`Сбой загрузки модели: ${error.message}`);
    }
  }

  onReset() {
    if (this.model) { this.model.dispose(); this.model = null; }
    this.dataLoader.dispose();
    this.trainData = null; this.testData = null;
    this.updateDataStatus(0,0);
    this.updateModelInfo();
    this.clearPreview();
    this.renderMetrics(null);
    this.showStatus('Сброс выполнен');
  }

  toggleVisor() { tfvis.visor().toggle(); }

  // ===================== Модель: простой автоэнкодер =====================
  createSimpleAutoencoder() {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({ inputShape:[28,28,1], filters:8, kernelSize:3, padding:'same', activation:'relu', name:'enc_conv1' }));
    model.add(tf.layers.conv2d({ filters:8, kernelSize:3, padding:'same', activation:'relu', name:'enc_conv2' }));
    model.add(tf.layers.conv2d({ filters:1, kernelSize:3, padding:'same', activation:'sigmoid', name:'dec_out' }));
    model.compile({ optimizer: tf.train.adam(1e-3), loss: 'meanSquaredError', metrics:['mse'] });
    return model;
  }

  // ===================== Рендер UI =====================
  updateDataStatus(trainCount, testCount) {
    document.getElementById('dataBadge').textContent = `Данные: ${trainCount} / ${testCount}`;
    document.getElementById('dataCounts').textContent = `${trainCount} / ${testCount}`;
  }

  updateModelInfo() {
    const infoEl = document.getElementById('modelInfo');
    const modelBadge = document.getElementById('modelBadge');
    if (!this.model) {
      infoEl.innerHTML = `<div class="muted">Нет модели</div><div></div>`;
      modelBadge.textContent = 'Модель: нет';
      return;
    }
    let params = 0;
    this.model.layers.forEach(l => l.getWeights().forEach(w => { params += w.size; }));
    infoEl.innerHTML = `
      <div>Слоёв</div><div>${this.model.layers.length}</div>
      <div>Параметров</div><div>${params.toLocaleString()}</div>
      <div>Оптимизатор</div><div>Adam (1e-3)</div>
      <div>Функция потерь</div><div>MSE</div>
    `;
    modelBadge.textContent = 'Модель: готова';
  }

  renderMetrics(stats) {
    const el = document.getElementById('metricsInfo');
    if (!stats) { el.innerHTML = `<div class="muted">Пока нет вычислений</div><div></div>`; return; }
    el.innerHTML = `
      <div>MSE (сред.)</div><div>${stats.mse.toFixed(6)}</div>
      <div>PSNR (дБ)</div><div>${stats.psnr.toFixed(2)}</div>
      <div>Образцов</div><div>${stats.n}</div>
    `;
  }

  clearPreview() { document.getElementById('previewContainer').innerHTML = ''; }

  renderDenoisingPreview(original, noisy, reconstructed, labels, indices) {
    const container = document.getElementById('previewContainer');
    container.innerHTML = '';
    const k = indices.length;
    const origArr = tf.unstack(original);
    const noisyArr = tf.unstack(noisy);
    const reconArr = tf.unstack(reconstructed);

    for (let i = 0; i < k; i++) {
      const card = document.createElement('div');
      card.className = 'card';
      const title = document.createElement('h4');
      title.textContent = `Сэмпл #${indices[i]} — Метка ${labels[i]}`;
      card.appendChild(title);

      const row = document.createElement('div'); row.className = 'triptych';

      const makeTile = (name, tens) => {
        const tile = document.createElement('div'); tile.className = 'tile';
        const canvas = document.createElement('canvas');
        this.dataLoader.draw28x28ToCanvas(tens.squeeze(), canvas, 4);
        const lab = document.createElement('label'); lab.textContent = name;
        tile.appendChild(canvas); tile.appendChild(lab);
        return tile;
      };

      row.appendChild(makeTile('Оригинал', origArr[i]));
      row.appendChild(makeTile('Шум', noisyArr[i]));
      row.appendChild(makeTile('Denoised', reconArr[i]));
      card.appendChild(row);

      const metrics = document.createElement('div'); metrics.className = 'metrics';
      metrics.textContent = 'Нажмите «Оценить» для сводных метрик (MSE/PSNR)';
      card.appendChild(metrics);

      container.appendChild(card);
    }

    // очистка временных тензоров (unstack возвращает новые)
    origArr.forEach(t => t.dispose());
    noisyArr.forEach(t => t.dispose());
    reconArr.forEach(t => t.dispose());
  }

  // ===================== Логи =====================
  showStatus(message) {
    const logs = document.getElementById('trainingLogs');
    const entry = document.createElement('div');
    entry.className = 'log-line';
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logs.appendChild(entry);
    logs.scrollTop = logs.scrollHeight;
  }

  showError(message) {
    const logs = document.getElementById('trainingLogs');
    const entry = document.createElement('div');
    entry.className = 'log-line log-err';
    entry.textContent = `[${new Date().toLocaleTimeString()}] ❌ ${message}`;
    logs.appendChild(entry);
    logs.scrollTop = logs.scrollHeight;
    console.error(message);
  }
}

// Инициализация
document.addEventListener('DOMContentLoaded', () => { new MNISTApp(); });
