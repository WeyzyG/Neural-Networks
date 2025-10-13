// data-loader.js — CSV loading, normalization, noise utils, AE helpers.

export async function loadTrainFromFiles(file) {
  const text = await readAsTextUTF8(file);
  const { labels, images } = parseCSV(text);
  const xs = toTensor01(images);
  const ys = oneHot(labels);
  return { xs, ys };
}

export async function loadTestFromFiles(file, options = {}) {
  const text = await readAsTextUTF8(file);
  const { labels, images } = parseCSV(text);
  let xs = toTensor01(images);
  const ys = oneHot(labels);
  const mode = options.noise ?? 'none';
  if (mode === 'gaussian') {
    const std = options.stdDev ?? 0.5;
    const noisy = addGaussianNoise01(xs, std);
    xs.dispose(); xs = noisy;
  } else if (mode === 'sp') {
    const p = options.prob ?? 0.1;
    const noisy = addSaltPepper01(xs, p);
    xs.dispose(); xs = noisy;
  }
  return { xs, ys };
}

export function splitTrainVal(xs, ys, valRatio = 0.1) {
  const n = xs.shape[0];
  const v = Math.floor(n * valRatio);
  const t = n - v;
  const trainXs = xs.slice([0,0,0,0], [t,28,28,1]);
  const trainYs = ys.slice([0,0], [t,10]);
  const valXs = xs.slice([t,0,0,0], [v,28,28,1]);
  const valYs = ys.slice([t,0], [v,10]);
  return { trainXs, trainYs, valXs, valYs };
}

export function getRandomTestBatch(xs, ys, k = 5) {
  const n = xs.shape[0];
  const idx = Array.from({length:k}, () => Math.floor(Math.random() * n));
  const batchXs = tf.gather(xs, idx);
  const batchYs = ys ? tf.gather(ys, idx) : null;
  return { batchXs, batchYs, idx };
}

export function draw28x28ToCanvas(t01, canvas, scale = 4) {
  const ctx = canvas.getContext('2d');
  const w = 28, h = 28;
  canvas.width = w * scale; canvas.height = h * scale;
  const img = new ImageData(w, h);
  const data = t01.reshape([w*h]).mul(255).clipByValue(0,255).toInt().dataSync();
  for (let i=0;i<w*h;i++){
    const v=data[i]; const o=i*4;
    img.data[o]=v; img.data[o+1]=v; img.data[o+2]=v; img.data[o+3]=255;
  }
  const tmp=document.createElement('canvas'); tmp.width=w; tmp.height=h;
  tmp.getContext('2d').putImageData(img,0,0);
  ctx.imageSmoothingEnabled=false; ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(tmp,0,0,w*scale,h*scale);
}

// -------- Noise utilities --------
export function addGaussianNoise01(xs01, stdDev=0.5){
  return tf.tidy(()=> xs01.add(tf.randomNormal(xs01.shape,0,stdDev)).clipByValue(0,1));
}
export function addSaltPepper01(xs01, p=0.1){
  return tf.tidy(()=>{
    const r=tf.randomUniform(xs01.shape);
    const z=r.less(p/2).toFloat();
    const o=r.greater(1-p/2).toFloat();
    const keep=tf.onesLike(z).sub(z).sub(o).clipByValue(0,1);
    return xs01.mul(keep).add(o).clipByValue(0,1);
  });
}

// -------- AE helper --------
export function makeNoisyCleanPairs(xsClean, mode='gaussian', args={stdDev:0.5, prob:0.1}){
  let noisy;
  if (mode==='gaussian') noisy=addGaussianNoise01(xsClean, args.stdDev??0.5);
  else if (mode==='sp') noisy=addSaltPepper01(xsClean, args.prob??0.1);
  else noisy=xsClean.clone();
  return { noisy, clean: xsClean };
}

// -------- Internal parsing --------
function parseCSV(text){
  const rows = text.split(/\r?\n/);
  const labels=[]; const images=[];
  for(const line of rows){
    if(!line) continue;
    const parts=line.split(',').filter(Boolean);
    if(parts.length!==785) continue;
    const label=parseInt(parts[0],10);
    const pixels=parts.slice(1).map(Number);
    labels.push(label); images.push(pixels);
  }
  return { labels, images };
}
function toTensor01(images){
  const flat=images.flat();
  const t=tf.tensor2d(flat,[images.length,784],'float32').div(255);
  return t.reshape([images.length,28,28,1]);
}
function oneHot(labels){
  return tf.tidy(()=> tf.oneHot(tf.tensor1d(labels,'int32'),10).toFloat());
}
function readAsTextUTF8(file){
  return new Promise((resolve,reject)=>{
    const r=new FileReader();
    r.onerror=()=>reject(r.error);
    r.onload=()=>resolve(r.result);
    r.readAsText(file,'utf-8');
  });
}
