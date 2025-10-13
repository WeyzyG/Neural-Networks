// data-loader.js
// CSV -> Tensors loader for MNIST with optional noise utilities.
// Reads local files only. No network fetches.

// ---- Parsing helpers ----
function parseCSVToArrays(text) {
  const rows = text.split(/\r?\n/);
  const labels = [];
  const images = [];
  for (const line of rows) {
    if (!line) continue;
    const parts = line.split(/[,\s]+/).filter(s => s.length > 0);
    if (parts.length !== 785) continue; // skip malformed rows
    const label = parseInt(parts[0], 10);
    const pixels = parts.slice(1).map(v => Number(v));
    labels.push(label);
    images.push(pixels);
  }
  return { labels, images };
}

// Normalize [0..255] -> [0..1]
function normalizeImagesFloat32(images2d) {
  const flat = images2d.flat();
  const xs = tf.tensor2d(flat, [images2d.length, 784], 'float32')
    .div(tf.scalar(255));
  return xs.reshape([images2d.length, 28, 28, 1]);
}

// One-hot labels depth 10
function oneHot(labels) {
  return tf.tidy(() => {
    const y = tf.tensor1d(labels, 'int32');
    const oh = tf.oneHot(y, 10).toFloat();
    return oh;
  });
}

// ---- Noise utilities (Step 1) ----
// Adds Gaussian noise with std=stdDev (default 0.5) and clips to [0,1].
export function addGaussianNoise01(xs01, stdDev = 0.5) {
  return tf.tidy(() => {
    const noise = tf.randomNormal(xs01.shape, 0, stdDev, 'float32');
    const noisy = xs01.add(noise).clipByValue(0, 1);
    return noisy;
  });
}

// Adds salt-and-pepper noise with probability p per pixel.
export function addSaltPepper01(xs01, p = 0.1) {
  return tf.tidy(() => {
    const rnd = tf.randomUniform(xs01.shape, 0, 1, 'float32');
    const salt = rnd.lessEqual(p / 2).toFloat();     // set to 0
    const pepper = rnd.greaterEqual(1 - p / 2).toFloat(); // set to 1
    const base = xs01.mul(tf.onesLike(salt).sub(salt)).add(pepper);
    return base.clipByValue(0, 1);
  });
}

// ---- Public API ----
export async function loadTrainFromFiles(file) {
  const text = await readAsTextUTF8(file);
  const { labels, images } = parseCSVToArrays(text);
  const xs = normalizeImagesFloat32(images);
  const ys = oneHot(labels);
  return { xs, ys };
}

export async function loadTestFromFiles(file, options = {}) {
  // options: { noise: 'none'|'gaussian'|'sp', stdDev?:number, prob?:number }
  const text = await readAsTextUTF8(file);
  const { labels, images } = parseCSVToArrays(text);
  let xs = normalizeImagesFloat32(images);
  const ys = oneHot(labels);

  const noiseMode = options.noise || 'none';
  if (noiseMode === 'gaussian') {
    const std = options.stdDev ?? 0.5;
    const noisy = addGaussianNoise01(xs, std);
    xs.dispose();
    xs = noisy;
  } else if (noiseMode === 'sp') {
    const p = options.prob ?? 0.1;
    const noisy = addSaltPepper01(xs, p);
    xs.dispose();
    xs = noisy;
  }
  return { xs, ys };
}

export function splitTrainVal(xs, ys, valRatio = 0.1) {
  const n = xs.shape[0];
  const valCount = Math.floor(n * valRatio);
  const trainCount = n - valCount;

  const trainXs = xs.slice([0, 0, 0, 0], [trainCount, 28, 28, 1]);
  const trainYs = ys.slice([0, 0], [trainCount, 10]);
  const valXs = xs.slice([trainCount, 0, 0, 0], [valCount, 28, 28, 1]);
  const valYs = ys.slice([trainCount, 0], [valCount, 10]);
  return { trainXs, trainYs, valXs, valYs };
}

export function getRandomTestBatch(xs, ys, k = 5) {
  const n = xs.shape[0];
  const idxs = [];
  for (let i = 0; i < k; i++) idxs.push(Math.floor(Math.random() * n));
  const batchXs = tf.gather(xs, idxs);
  const batchYs = tf.gather(ys, idxs);
  return { batchXs, batchYs, idxs };
}

export function draw28x28ToCanvas(tensor01, canvas, scale = 4) {
  const [h, w] = [28, 28];
  const data = tensor01.reshape([h, w]).mul(255).toInt();
  const ctx = canvas.getContext('2d');
  canvas.width = w * scale;
  canvas.height = h * scale;
  const imageData = ctx.createImageData(w, h);
  const arr = data.dataSync();
  for (let i = 0; i < w * h; i++) {
    const v = arr[i];
    imageData.data[i * 4 + 0] = v;
    imageData.data[i * 4 + 1] = v;
    imageData.data[i * 4 + 2] = v;
    imageData.data[i * 4 + 3] = 255;
  }
  // draw native size then scale up
  const tmp = document.createElement('canvas');
  tmp.width = w;
  tmp.height = h;
  tmp.getContext('2d').putImageData(imageData, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(tmp, 0, 0, w * scale, h * scale);
  data.dispose();
}

// ---- File reader ----
function readAsTextUTF8(file) {
  return new Promise((resolve, reject) => {
    try {
      const reader = new FileReader();
      reader.onerror = () => reject(reader.error);
      reader.onload = () => resolve(reader.result);
      reader.readAsText(file, 'utf-8');
    } catch (e) {
      reject(e);
    }
  });
}
