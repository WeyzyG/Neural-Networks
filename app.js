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

  // --- EDA SECTION ---
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
    classDistDiv.innerHTML = '<b>Class Distribution (Train)</b>';

    const trDist = this.getClassDist(this.trainData.labels);
    // TFJS-VIS BAR CHART
    const barchartData = FASHION_LABELS.map((lbl, i) => ({
      index: lbl,
      value: trDist[i]
    }));
    tfvis.render.barchart(
      { name: 'Class Distribution Chart', tab: 'EDA', styles: { height: '250px' } },
      barchartData,
      { xLabel: 'Class', yLabel: 'Count', height: 250 }
    );
    // Встраиваем созданный барчарт в твой div, если surface появился
    setTimeout(() => {
      const surfaces = document.querySelectorAll('[data-surface-name="Class Distribution Chart"]');
      if (surfaces.length) {
        classDistDiv.appendChild(surfaces[0]);
      }
    }, 400);

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

  // --- остальные методы: train, evaluate, preview и прочее ---
  // (оставь неизменными, кроме вызова runEDA)
  // ...
}

document.addEventListener('DOMContentLoaded', () => {
  new MNISTApp();
});
