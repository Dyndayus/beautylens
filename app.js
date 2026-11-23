const MODEL_PATH = './model_makeup_new.onnx';
const CLASS_NAMES = ['Bedak', 'Brush', 'Lippies'];
let session = null;
let webcamStream = null;
let minLogitThreshold = 3.8;

// Generate cahaya kelip-kelip otomatis di semua halaman
function createSparkles() {
  const layer = document.getElementById('bg-decor-layer');
  if (!layer) return;

  for (let i = 0; i < 28; i++) {
    const sp = document.createElement('div');
    sp.className = 'sparkle';
    sp.style.left = `${Math.random() * 100}%`;
    sp.style.top = `${Math.random() * 100}%`;
    sp.style.setProperty('--dur', `${2.2 + Math.random() * 3}s`);
    sp.style.setProperty('--delay', `${Math.random() * 3}s`);
    const size = 6 + Math.random() * 8;
    sp.style.width = `${size}px`;
    sp.style.height = `${size}px`;
    layer.appendChild(sp);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  createSparkles();

  // Cek apakah elemen halaman Home ada (agar tidak error saat membuka About)
  const statusText = document.getElementById('status-text');
  if (!statusText) return;

  const statusDot = document.getElementById('status-dot');

  if (window.location.protocol === 'file:') {
    statusText.textContent = 'Buka lewat VS Code "Open with Live Server" agar AI aktif!';
    statusDot.style.background = '#e11d48';
  }

  document.getElementById('btn-tab-upload').addEventListener('click', () => switchTab('upload'));
  document.getElementById('btn-tab-camera').addEventListener('click', () => switchTab('camera'));
  document.getElementById('file-input').addEventListener('change', handleImageUpload);
  
  // Tombol Kontrol Kamera
  document.getElementById('btn-start-cam').addEventListener('click', startCamera);
  document.getElementById('btn-capture-cam').addEventListener('click', captureAndAnalyze);
  document.getElementById('btn-retake-cam').addEventListener('click', retakeCamera);
  document.getElementById('btn-stop-cam').addEventListener('click', stopCamera);

  document.getElementById('strict-slider').addEventListener('input', (e) => {
    minLogitThreshold = parseFloat(e.target.value);
    document.getElementById('strict-val').textContent = `Logit ≥ ${minLogitThreshold.toFixed(1)}`;
  });

  initModel();
});

async function initModel() {
  const statusText = document.getElementById('status-text');
  const statusDot = document.getElementById('status-dot');

  if (typeof ort === 'undefined') {
    statusText.textContent = 'Gagal memuat ONNX Runtime. Cek koneksi internet.';
    statusDot.style.background = '#e11d48';
    return;
  }

  try {
    session = await ort.InferenceSession.create(MODEL_PATH);
    statusText.textContent = 'Model Siap Digunakan';
    statusDot.style.background = '#10b981';
    statusDot.style.animation = 'none';
  } catch (err) {
    statusText.textContent = 'Model tidak ditemukan. Pastikan buka via Live Server!';
    statusDot.style.background = '#e11d48';
    console.error(err);
  }
}

function switchTab(mode) {
  const btnUpload = document.getElementById('btn-tab-upload');
  const btnCamera = document.getElementById('btn-tab-camera');
  const panelUpload = document.getElementById('panel-upload');
  const panelCamera = document.getElementById('panel-camera');

  if (mode === 'upload') {
    btnUpload.classList.add('active');
    btnCamera.classList.remove('active');
    panelUpload.classList.add('active');
    panelCamera.classList.remove('active');
    stopCamera();
  } else {
    btnCamera.classList.add('active');
    btnUpload.classList.remove('active');
    panelCamera.classList.add('active');
    panelUpload.classList.remove('active');
  }
}

function imageToTensor(sourceElement) {
  const canvas = document.getElementById('hidden-canvas');
  const ctx = canvas.getContext('2d');
  ctx.drawImage(sourceElement, 0, 0, 224, 224);

  const imgData = ctx.getImageData(0, 0, 224, 224).data;
  const floatData = new Float32Array(3 * 224 * 224);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < 224 * 224; i++) {
    const r = imgData[i * 4] / 255.0;
    const g = imgData[i * 4 + 1] / 255.0;
    const b = imgData[i * 4 + 2] / 255.0;

    floatData[i] = (r - mean[0]) / std[0];
    floatData[224 * 224 + i] = (g - mean[1]) / std[1];
    floatData[2 * 224 * 224 + i] = (b - mean[2]) / std[2];
  }

  return new ort.Tensor('float32', floatData, [1, 3, 224, 224]);
}

function calibratedProbs(logits) {
  return logits.map(logit => 1 / (1 + Math.exp(-(logit - 2.2))));
}

async function predict(sourceElement) {
  if (!session) {
    alert('Model AI belum siap. Pastikan dibuka lewat Live Server!');
    return;
  }

  const inputTensor = imageToTensor(sourceElement);
  const feeds = { [session.inputNames[0]]: inputTensor };
  const results = await session.run(feeds);
  const logits = Array.from(results[session.outputNames[0]].data);

  const probs = calibratedProbs(logits);
  const maxIdx = logits.indexOf(Math.max(...logits));
  const maxProb = probs[maxIdx] * 100;
  const maxLogit = logits[maxIdx];

  const sortedLogits = [...logits].sort((a, b) => b - a);
  const logitMargin = sortedLogits[0] - sortedLogits[1];

  const isMakeupValid = (maxLogit >= minLogitThreshold) && (logitMargin >= 2.2);

  document.getElementById('empty-state').style.display = 'none';
  document.getElementById('result-content').style.display = 'block';

  const verdictBox = document.getElementById('verdict-box');
  const verdictLabel = document.getElementById('verdict-label');
  const predTitle = document.getElementById('pred-title');
  const predNote = document.getElementById('pred-note');

  if (isMakeupValid) {
    verdictBox.className = 'verdict-box verdict-valid';
    verdictLabel.textContent = '✅ Produk Makeup Terdeteksi';
    predTitle.textContent = `${CLASS_NAMES[maxIdx]} (${maxProb.toFixed(1)}%)`;
    predNote.textContent = `Raw Logit: ${maxLogit.toFixed(2)} • Selisih Margin: ${logitMargin.toFixed(2)}`;
  } else {
    verdictBox.className = 'verdict-box verdict-unknown';
    verdictLabel.textContent = '⚠️ Di Luar Kategori Makeup';
    predTitle.textContent = `Objek Tidak Dikenali`;
    predNote.textContent = `Skor keyakinan di bawah ambang batas aman (Logit tertinggi: ${maxLogit.toFixed(2)})`;
  }

  probs.forEach((p, idx) => {
    const pct = (p * 100).toFixed(1);
    document.getElementById(`prob-${idx}`).textContent = `${pct}%`;
    const barEl = document.getElementById(`bar-${idx}`);
    barEl.style.width = `${pct}%`;
    barEl.style.opacity = isMakeupValid ? '1' : '0.45';
  });
}

function handleImageUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const imgEl = document.getElementById('preview-img');
  document.getElementById('upload-placeholder').style.display = 'none';
  document.getElementById('reupload-hint').style.display = 'block';

  imgEl.src = URL.createObjectURL(file);
  imgEl.style.display = 'block';
  imgEl.onload = () => predict(imgEl);
}

// 1. Menyalakan preview kamera
async function startCamera() {
  const video = document.getElementById('webcam');
  const snapshotImg = document.getElementById('cam-snapshot');
  const camPlaceholder = document.getElementById('cam-placeholder');

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    video.srcObject = webcamStream;
    video.style.display = 'block';
    snapshotImg.style.display = 'none';
    camPlaceholder.style.display = 'none';

    // Atur tampilan tombol
    document.getElementById('btn-start-cam').style.display = 'none';
    document.getElementById('btn-capture-cam').style.display = 'inline-block';
    document.getElementById('btn-retake-cam').style.display = 'none';
  } catch (err) {
    alert('Gagal mengakses kamera: ' + err.message);
  }
}

// 2. Mengambil gambar (jepret) dari video dan menganalisanya
async function captureAndAnalyze() {
  const video = document.getElementById('webcam');
  const snapshotImg = document.getElementById('cam-snapshot');
  if (!webcamStream || video.readyState < 2) return;

  // Tangkap frame video saat ini ke canvas resolusi penuh
  const snapCanvas = document.createElement('canvas');
  snapCanvas.width = video.videoWidth || 640;
  snapCanvas.height = video.videoHeight || 480;
  snapCanvas.getContext('2d').drawImage(video, 0, 0, snapCanvas.width, snapCanvas.height);

  // Tampilkan hasil jepretan sebagai foto diam dan sembunyikan live video
  snapshotImg.src = snapCanvas.toDataURL('image/png');
  snapshotImg.style.display = 'block';
  video.style.display = 'none';

  // Ubah tombol ke mode "Foto Ulang"
  document.getElementById('btn-capture-cam').style.display = 'none';
  document.getElementById('btn-retake-cam').style.display = 'inline-block';

  // Jalankan analisa AI pada gambar yang dijepret
  snapshotImg.onload = () => predict(snapshotImg);
}

// 3. Kembali ke tampilan kamera live untuk foto ulang
function retakeCamera() {
  const video = document.getElementById('webcam');
  const snapshotImg = document.getElementById('cam-snapshot');

  snapshotImg.style.display = 'none';
  video.style.display = 'block';

  document.getElementById('btn-retake-cam').style.display = 'none';
  document.getElementById('btn-capture-cam').style.display = 'inline-block';
}

// 4. Mematikan kamera sepenuhnya
function stopCamera() {
  const video = document.getElementById('webcam');
  const snapshotImg = document.getElementById('cam-snapshot');
  const camPlaceholder = document.getElementById('cam-placeholder');

  if (camPlaceholder) camPlaceholder.style.display = 'block';
  if (video) video.style.display = 'block';
  if (snapshotImg) snapshotImg.style.display = 'none';

  document.getElementById('btn-start-cam').style.display = 'inline-block';
  document.getElementById('btn-capture-cam').style.display = 'none';
  document.getElementById('btn-retake-cam').style.display = 'none';

  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
}