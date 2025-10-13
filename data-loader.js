class MNISTDataLoader {
  constructor() {
    this.trainData = null;
    this.testData = null;
  }

  // Парсинг CSV: label + 784 пикселей (28x28), нормализация 0..1
  async loadCSVFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const content = event.target.result;
          const lines = content.split('\n').filter(line => line.trim() !== '');
          const labels = [];
          const pixels = [];
          for (const line of lines) {
            const values = line.split(',').map(Number);
            if (values.length !== 785) continue;
            labels.push(values[0]);
            pixels.push(values.slice(1));
          }
          if (labels.length === 0) {
            reject(new Error('No valid data found in file'));
            return;
          }
          const xs = tf.tidy(() => tf.tensor2d(pixels).div(255).reshape([labels.length, 28, 28, 1]));
          const ys = tf.tidy(() => tf.oneHot(labels, 10));
          resolve({ xs, ys, count: labels.length, labels });
        } catch (error) { reject(error); }
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  }

  async loadTrainFromFiles(file) {
    this.trainData = await this.loadCSVFile(file);
    return this.trainData;
  }

  async loadTestFromFiles(file) {
    this.testData = await this.loadCSVFile(file);
    return this.testData;
  }

  // Гауссов шум
  addNoise(images, noiseStd = 0.1) {
    return tf.tidy(() => {
      const noise = tf.randomNormal(images.shape, 0, noiseStd);
      return images.add(noise).clipByValue(0, 1);
    });
  }

  // Разбиение на батчи (если понадобится)
  splitTrainVal(xs, ys, valRatio = 0.1) {
    return tf.tidy(() => {
      const numVal = Math.floor(xs.shape[0] * valRatio);
      const numTrain = xs.shape[0] - numVal;
      const trainXs = xs.slice([0, 0, 0, 0], [numTrain, 28, 28, 1]);
      const trainYs = ys.slice([0, 0], [numTrain, 10]);
      const valXs = xs.slice([numTrain, 0, 0, 0], [numVal, 28, 28, 1]);
      const valYs = ys.slice([numTrain, 0], [numVal, 10]);
      return { trainXs, trainYs, valXs, valYs };
    });
  }

  // Случайный батч для предпросмотра
  getRandomTestBatch(xs, ys, k = 5) {
    return tf.tidy(() => {
      const shuffled = tf.util.createShuffledIndices(xs.shape[0]);
      const idx = Array.from(shuffled.slice(0, k));
      const batchXs = tf.gather(xs, idx);
      const batchYs = tf.gather(ys, idx);
      return { batchXs, batchYs, indices: idx };
    });
  }

  // Рисование 28x28 на canvas
  draw28x28ToCanvas(tensor, canvas, scale = 4) {
    const ctx = canvas.getContext('2d');
    const img = new ImageData(28, 28);
    const data = tensor.reshape([28, 28]).mul(255).dataSync();
    for (let i = 0; i < 784; i++) {
      const v = data[i];
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    const temp = document.createElement('canvas');
    temp.width = 28; temp.height = 28;
    const tctx = temp.getContext('2d');
    tctx.putImageData(img, 0, 0);

    canvas.width = 28 * scale;
    canvas.height = 28 * scale;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(temp, 0, 0, 28 * scale, 28 * scale);
  }

  // PSNR (максимум интенсивности = 1)
  calculatePSNR(original, reconstructed) {
    return tf.tidy(() => {
      const mse = reconstructed.sub(original).square().mean();
      const psnr = tf.scalar(10).mul(tf.scalar(1).div(mse).log().div(tf.scalar(Math.log(10))));
      return { mse, psnr }; // вернуть тензоры (caller сам .data())
    });
  }

  dispose() {
    if (this.trainData) {
      this.trainData.xs.dispose(); this.trainData.ys.dispose();
      this.trainData = null;
    }
    if (this.testData) {
      this.testData.xs.dispose(); this.testData.ys.dispose();
      this.testData = null;
    }
  }
}
