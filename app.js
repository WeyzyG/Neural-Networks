// app.js — CNN classifier + Autoencoder with text logs in console style.

import {
  loadTrainFromFiles, loadTestFromFiles, splitTrainVal,
  getRandomTestBatch, draw28x28ToCanvas, makeNoisyCleanPairs
} from './data-loader.js';

const els = {
  inputTrain: document.getElementById('trainCsv'),
  inputTest: document.getElementById('testCsv'),
  btnLoad: document.getElementById('btnLoad'),
  btnTrain: document.getElementById('btnTrain'),
  btnEval: document.getElementById('btnEval'),
  btnTest5: document.getElementById('btnTest5'),
  btnSave: document.getElementById('btnSave'),
  btnLoadModel: document.getElementById('btnLoadModel'),
  btnReset: document.getElementById('btnReset'),
  btnVisor: document.getElementById('btnVisor'),
  noiseMode: document.getElementById('noiseMode'),
  noiseStd: document.getElementById('noiseStd'),
  noiseProb: document.getElementById('noiseProb'),
  btnBuildAE: document.getElementById('btnBuildAE'),
  btnTrainAE: document.getElementById('btnTrainAE'),
  btnEvalAE: document.getElementById('btnEvalAE'),
  btnTest5AE: document.getElementById('btnTest5AE'),
  btnSaveAE: document.getElementById('btnSaveAE'),
  dataStatus: document.getElementById('dataStatus'),
  modelInfo: document.getElementById('modelInfo'),
  previewStrip: document.getElementById('previewStrip'),
  previewLabels: document.getElementById('previewLabels'),
  textLogs: document.getElementById('textLogs')
};

// -------- Text logger with timestamp and styles --------
function time() {
  const d = new Date();
  const pad = n => String(n).padStart(2,'0');
  return `[${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}]`;
}
function appendLine(html) {
  els.textLogs.insertAdjacentHTML('beforeend', html + '\n');
  els.textLogs.scrollTop = els.textLogs.scrollHeight;
}
function line(msg){ appendLine(`${time()} ${msg}`); }
function ok(msg){ appendLine(`${time()} <span class="ok">✓ ${escapeHtml(msg)}</span>`); }
function err(msg){ appendLine(`${time()} <span class="err">✗ ${escapeHtml(msg)}</span>`); }
function warn(msg){ appendLine(`${time()} <span class="warn">! ${escapeHtml(msg)}</span>`); }
function section(title){ appendLine(`${time()} <span class="sec">=== ${escapeHtml(title)} ===</span>`); }
function escapeHtml(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function clearLogs(){ els.textLogs.textContent=''; }

let state = {
  files: { train: null, test: null },
  cls: { model: null, train: null, testClean: null, testNoisy: null },
  ae: { model: null },
  noise: { mode: 'gaussian', std: 0.5, prob: 0.1 }
};

// Buttons
els.inputTrain.addEventListener('change', e => state.files.train = e.target.files[0]);
els.inputTest.addEventListener('change', e => state.files.test = e.target.files[0]);
els.btnVisor.addEventListener('click', () => tfvis.visor().toggle());
els.btnLoad.addEventListener('click', onLoadData);
els.btnTrain.addEventListener('click', onTrainClassifier);
els.btnEval.addEventListener('click', onEvaluateClassifier);
els.btnTest5.addEventListener('click', onTest5Classifier);
els.btnSave.addEventListener('click', () => state.cls.model?.save('downloads://mnist-cnn'));
els.btnLoadModel.addEventListener('click', onLoadClassifierFiles);
els.btnReset.addEventListener('click', () => onReset(false));
els.noiseMode.addEventListener('change', async () => { state.noise.mode = els.noiseMode.value; await rebuildNoisyTest('Noise mode changed'); });
els.noiseStd.addEventListener('input', async () => { state.noise.std = parseFloat(els.noiseStd.value); if (state.noise.mode==='gaussian') await rebuildNoisyTest('Noise std changed'); });
els.noiseProb.addEventListener('input', async () => { state.noise.prob = parseFloat(els.noiseProb.value); if (state.noise.mode==='sp') await rebuildNoisyTest('Noise prob changed'); });

// Busy UI helper
const allButtons = [
  'btnLoad','btnTrain','btnEval','btnTest5','btnSave','btnLoadModel','btnReset','btnVisor',
  'btnBuildAE','btnTrainAE','btnEvalAE','btnTest5AE','btnSaveAE'
].map(id => document.getElementById(id));
function setWorking(busy){
  allButtons.forEach(b => b && (b.disabled = busy && b.id !== 'btnVisor'));
  document.body.style.cursor = busy ? 'progress' : 'default';
}

// ---------- Data load ----------
async function onLoadData(){
  try{
    clearLogs();
    if(!state.files.train || !state.files.test){ warn('Select both train and test CSV first'); return; }
    setWorking(true);
    line('Loading training data...');
    state.cls.train = await loadTrainFromFiles(state.files.train);
    line('Loading test data...');
    state.cls.testClean = await loadTestFromFiles(state.files.test, { noise: 'none' });
    await rebuildNoisyTest('Test noisy cache built');

    ok('Data loaded successfully!');
    const tr = state.cls.train.xs.shape; const te = state.cls.testClean.xs.shape;
    line(`Data check - Train shape: ${tr.join(',')}, Test shape: ${te.join(',')}`);
    const minmax = await getRange(state.cls.train.xs);
    line(`Data range - Min: ${minmax.min.toFixed(3)}, Max: ${minmax.max.toFixed(3)}`);

    els.dataStatus.textContent = `Train: ${tr[0]} | Test: ${te[0]} (noisy mode=${state.noise.mode})`;
  }catch(e){
    console.error(e); err('Failed to load data'); warn(String(e.message||e));
  }finally{
    setWorking(false);
  }
}

async function rebuildNoisyTest(reason=''){
  if(!state.files.test) return;
  state.cls.testNoisy?.xs?.dispose(); state.cls.testNoisy?.ys?.dispose();
  state.cls.testNoisy = await loadTestFromFiles(state.files.test, {
    noise: state.noise.mode, stdDev: state.noise.std, prob: state.noise.prob
  });
  if (reason) line(`${reason}; noisy mode=${state.noise.mode}`);
}

async function getRange(t){
  const d = await t.data();
  let mn=Infinity, mx=-Infinity;
  for (let i=0;i<d.length;i++){ const v=d[i]; if(v<mn) mn=v; if(v>mx) mx=v; }
  return {min:mn, max:mx};
}

// ---------- Classifier ----------
function buildClassifier(){
  const m=tf.sequential();
  m.add(tf.layers.conv2d({filters:32,kernelSize:3,activation:'relu',padding:'same',inputShape:[28,28,1]}));
  m.add(tf.layers.conv2d({filters:64,kernelSize:3,activation:'relu',padding:'same'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.dropout({rate:0.25}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({units:128,activation:'relu'}));
  m.add(tf.layers.dropout({rate:0.5}));
  m.add(tf.layers.dense({units:10,activation:'softmax'}));
  m.compile({optimizer:'adam',loss:'categoricalCrossentropy',metrics:['accuracy']});
  return m;
}

async function onTrainClassifier(){
  try{
    if(!state.cls.train) { warn('Load data first'); return; }
    setWorking(true);
    section('Classifier training started');
    state.cls.model?.dispose();
    state.cls.model = buildClassifier();

    const {trainXs, trainYs, valXs, valYs} = splitTrainVal(state.cls.train.xs, state.cls.train.ys, 0.1);
    const epochs=8; const surface={name:'Classifier Training', tab:'Training Logs'};
    const metrics=['loss','val_loss','acc','val_acc'];
    const uiCb = progressCallbacks('Classifier', epochs);

    const t0=performance.now();
    await state.cls.model.fit(trainXs, trainYs, {
      epochs, batchSize:128, shuffle:true, validationData:[valXs,valYs],
      callbacks:[tfvis.show.fitCallbacks(surface, metrics), uiCb]
    });
    const dur=((performance.now()-t0)/1000).toFixed(1);
    ok(`Classifier training finished in ${dur}s`);

    trainXs.dispose(); trainYs.dispose(); valXs.dispose(); valYs.dispose();
    renderModelSummary(state.cls.model, 'Classifier');
  }catch(e){
    console.error(e); err('Training error'); warn(String(e.message||e));
  }finally{
    setWorking(false);
  }
}

function progressCallbacks(prefix, totalEpochs){
  let ep=0;
  return {
    onEpochBegin: async (epoch)=>{ ep=epoch; line(`${prefix}: epoch ${epoch+1}/${totalEpochs}...`); await tf.nextFrame(); },
    onBatchEnd: async (batch, logs)=>{ line(`${prefix}: epoch ${ep+1}/${totalEpochs}, batch ${batch}, loss=${Number(logs.loss).toFixed(6)}`); await tf.nextFrame(); }
  };
}

async function onEvaluateClassifier(){
  try{
    if(!state.cls.model || !state.cls.testClean){ warn('Need model and data'); return; }
    setWorking(true);
    section('Classifier evaluation');
    const use = state.cls.testNoisy ?? state.cls.testClean;
    const res = await state.cls.model.evaluate(use.xs, use.ys, { batchSize: 256 });
    const acc = Array.isArray(res) ? res[1].dataSync()[0] : res.dataSync()[0];
    ok(`Accuracy (noisy=${state.noise.mode}) = ${(acc*100).toFixed(2)}%`);

    const preds = state.cls.model.predict(use.xs).argMax(-1);
    const labels = use.ys.argMax(-1);
    const cm = await tfvis.metrics.confusionMatrix(labels, preds, 10);
    await tfvis.render.confusionMatrix({ name:'Confusion Matrix', tab:'Metrics' }, { values: cm, tickLabels:[...'0123456789'] });
    const pca = tfvis.metrics.perClassAccuracy(labels, preds);
    await tfvis.render.barchart({ name:'Per‑class Accuracy', tab:'Metrics' }, Object.keys(pca).map(k=>({x:k,y:pca[k]})));
    preds.dispose(); labels.dispose();
  }catch(e){
    console.error(e); err('Evaluation error'); warn(String(e.message||e));
  }finally{
    setWorking(false);
  }
}

async function onTest5Classifier(){
  try{
    if(!state.cls.model || !state.cls.testClean){ warn('Need model and data'); return; }
    setWorking(true);
    section('Classifier 5‑sample preview');
    const use = state.cls.testNoisy ?? state.cls.testClean;
    els.previewStrip.innerHTML=''; els.previewLabels.innerHTML='';

    const { batchXs, batchYs } = getRandomTestBatch(use.xs, use.ys, 5);
    const preds = state.cls.model.predict(batchXs).argMax(-1);
    const yTrue = batchYs.argMax(-1);

    const pa = Array.from(preds.dataSync()); const ya = Array.from(yTrue.dataSync());
    for (const img of tf.unstack(batchXs)) {
      const c = document.createElement('canvas'); els.previewStrip.appendChild(c);
      draw28x28ToCanvas(img, c, 4); img.dispose();
    }
    for (let i=0;i<pa.length;i++){
      const s=document.createElement('span'); s.textContent=String(pa[i]);
      s.style.marginRight='18px'; s.style.color= pa[i]===ya[i] ? '#2e7d32' : '#c0392b';
      els.previewLabels.appendChild(s);
    }
    ok('Preview updated');
    preds.dispose(); yTrue.dispose(); batchXs.dispose(); batchYs.dispose();
  }catch(e){
    console.error(e); err('Preview error'); warn(String(e.message||e));
  }finally{
    setWorking(false);
  }
}

async function onLoadClassifierFiles(){
  try{
    const j = document.getElementById('modelJson').files[0];
    const b = document.getElementById('modelBin').files[0];
    if(!j || !b){ warn('Select model.json and weights.bin'); return; }
    setWorking(true);
    line('Loading model from files...');
    state.cls.model?.dispose();
    state.cls.model = await tf.loadLayersModel(tf.io.browserFiles([j,b]));
    ok('Model loaded successfully!');
    renderModelSummary(state.cls.model, 'Classifier (loaded)');
  }catch(e){
    console.error(e); err('Model load error'); warn(String(e.message||e));
  }finally{
    setWorking(false);
  }
}

// ---------- Autoencoder ----------
function buildAE(){
  state.ae.model?.dispose();
  const m = tf.sequential();
  // Encoder
  m.add(tf.layers.conv2d({filters:32,kernelSize:3,activation:'relu',padding:'same',inputShape:[28,28,1]}));
  m.add(tf.layers.maxPooling2d({poolSize:2,padding:'same'}));
  m.add(tf.layers.conv2d({filters:64,kernelSize:3,activation:'relu',padding:'same'}));
  m.add(tf.layers.maxPooling2d({poolSize:2,padding:'same'})); // 7x7
  // Decoder
  m.add(tf.layers.conv2dTranspose({filters:64,kernelSize:3,strides:2,activation:'relu',padding:'same'})); // 14x14
  m.add(tf.layers.conv2dTranspose({filters:32,kernelSize:3,strides:2,activation:'relu',padding:'same'})); // 28x28
  m.add(tf.layers.conv2d({filters:1,kernelSize:3,activation:'sigmoid',padding:'same'}));
  m.compile({optimizer:'adam', loss:'meanSquaredError'});
  state.ae.model = m;
  ok('Autoencoder built');
}

els.btnBuildAE.addEventListener('click', buildAE);
els.btnTrainAE.addEventListener('click', onTrainAE);
els.btnEvalAE.addEventListener('click', onEvalAE);
els.btnTest5AE.addEventListener('click', onTest5AE);
els.btnSaveAE.addEventListener('click', ()=> state.ae.model?.save('downloads://mnist-ae'));

async function onTrainAE(){
  try{
    if(!state.cls.train){ warn('Load data first'); return; }
    if(!state.ae.model) buildAE();
    setWorking(true);
    section('Starting REAL training (AE)');
    const { noisy, clean } = makeNoisyCleanPairs(state.cls.train.xs, state.noise.mode, { stdDev: state.noise.std, prob: state.noise.prob });

    // Quick sanity check
    const quick = await clean.mean().data();
    line(`Quick test - Avg output: ${Number(quick[0]).toFixed(3)}`);

    const epochs=8;
    const uiCb = progressCallbacks('AE', epochs);
    const t0=performance.now();
    await state.ae.model.fit(noisy, clean, { epochs, batchSize:128, shuffle:true, validationSplit:0.1, callbacks:[uiCb] });
    const dur=((performance.now()-t0)/1000).toFixed(1);
    ok(`AE training finished in ${dur}s`);

    noisy.dispose(); // clean = train.xs reference: не освобождаем
  }catch(e){
    console.error(e); err('AE train error'); warn(String(e.message||e));
  }finally{
    setWorking(false);
  }
}

async function onEvalAE(){
  try{
    if(!state.ae.model || !state.cls.testClean){ warn('Need AE and data'); return; }
    setWorking(true);
    section('Evaluating model performance (AE)');
    const { noisy } = makeNoisyCleanPairs(state.cls.testClean.xs, state.noise.mode, { stdDev: state.noise.std, prob: state.noise.prob });
    const res = await state.ae.model.evaluate(noisy, state.cls.testClean.xs, { batchSize: 256 });
    const mse = Array.isArray(res) ? res[0].dataSync()[0] : res.dataSync()[0];
    line('=== Detailed Analysis ===');
    ok(`MSE: ${mse.toFixed(6)}`);
    noisy.dispose();
  }catch(e){
    console.error(e); err('AE eval error'); warn(String(e.message||e));
  }finally{
    setWorking(false);
  }
}

async function onTest5AE(){
  try{
    if(!state.ae.model || !state.cls.testClean){ warn('Need AE and data'); return; }
    setWorking(true);
    section('Denoising preview (5 samples)');
    els.previewStrip.innerHTML=''; els.previewLabels.innerHTML='';
    const { noisy } = makeNoisyCleanPairs(state.cls.testClean.xs, state.noise.mode, { stdDev: state.noise.std, prob: state.noise.prob });
    const { batchXs } = getRandomTestBatch(noisy, null, 5);
    const den = state.ae.model.predict(batchXs);

    // top row: noisy
    const k=batchXs.shape[0];
    for(let i=0;i<k;i++){
      const c=document.createElement('canvas'); els.previewStrip.appendChild(c);
      draw28x28ToCanvas(batchXs.slice([i,0,0,0],[1,28,28,1]).squeeze(), c, 4);
    }
    // line break
    const br=document.createElement('div'); br.style.flexBasis='100%'; els.previewStrip.appendChild(br);
    // bottom row: denoised
    for(let i=0;i<k;i++){
      const c=document.createElement('canvas'); els.previewStrip.appendChild(c);
      draw28x28ToCanvas(den.slice([i,0,0,0],[1,28,28,1]).squeeze(), c, 4);
    }
    ok('Preview updated (top: noisy, bottom: denoised)');
    batchXs.dispose(); den.dispose(); noisy.dispose();
  }catch(e){
    console.error(e); err('AE preview error'); warn(String(e.message||e));
  }finally{
    setWorking(false);
  }
}

// ---------- Common ----------
function renderModelSummary(model, title){
  els.modelInfo.innerHTML='';
  const div=document.createElement('div');
  div.textContent=`${title} — layers: ${model.layers.length}, params: ${model.countParams()}`;
  els.modelInfo.appendChild(div);
  model.summary(undefined, undefined, line => {
    const el=document.createElement('div'); el.className='muted'; el.textContent=line; els.modelInfo.appendChild(el);
  });
}

async function onReset(keep=false){
  setWorking(true);
  clearLogs();
  line('Resetting...');
  els.previewStrip.innerHTML=''; els.previewLabels.innerHTML='';
  els.modelInfo.textContent='No model loaded';
  state.cls.model?.dispose(); state.cls.model=null;
  state.ae.model?.dispose(); state.ae.model=null;
  for (const k of ['train','testClean','testNoisy']){
    if(state.cls[k]?.xs) state.cls[k].xs.dispose();
    if(state.cls[k]?.ys) state.cls[k].ys.dispose();
    state.cls[k]=null;
  }
  if(!keep){ els.inputTrain.value=''; els.inputTest.value=''; state.files.train=null; state.files.test=null; }
  ok('Reset done');
  setWorking(false);
}
