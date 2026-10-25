// Rust detector: runs the YOLOv10s corrosion model in the browser with onnxruntime-web.
(() => {
  'use strict';

  const MODEL_URL = 'model/best.onnx';
  const MODEL_BYTES = 29142788;   // fallback for the progress bar if the server hides the size
  const SIZE = 640;               // model input size
  const IOU = 0.7;                // same NMS threshold as Ultralytics predict
  const FLOOR = 0.05;             // lowest confidence the slider can reach
  const MAX_DET = 300;
  const BOX = '#19d3e6';

  const $ = (id) => document.getElementById(id);
  const stage = $('stage'), dropzone = $('dropzone'), viewer = $('viewer');
  const canvas = $('canvas'), ctx = canvas.getContext('2d');
  const fileInput = $('file'), chooseBtn = $('choose'), scanline = $('scanline');
  const statusEl = $('status'), statusText = $('status-text');
  const conf = $('conf'), confOut = $('conf-out');
  const results = $('results'), countEl = $('count'), metaEl = $('meta');
  const list = $('det-list'), downloadBtn = $('download');

  let session = null;
  let sessionReady = null;
  let image = null;        // ImageBitmap of the current photo
  let allDets = [];        // detections at FLOOR confidence, after NMS, in image pixels
  let active = -1;         // detection highlighted from the list
  let lastMs = 0;
  let busy = false;

  ort.env.wasm.wasmPaths = new URL('ort/', document.baseURI).href;

  function setStatus(text, state) {
    statusText.textContent = text;
    statusEl.className = 'status' + (state ? ' ' + state : '');
  }

  // ---------- model loading ----------
  async function fetchModel() {
    const res = await fetch(MODEL_URL);
    if (!res.ok) throw new Error('Model file not found (' + res.status + ')');
    const total = Number(res.headers.get('content-length')) || MODEL_BYTES;
    if (!res.body || !res.body.getReader) return new Uint8Array(await res.arrayBuffer());
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      const mb = (n) => (n / 1048576).toFixed(0);
      setStatus('Loading model… ' + mb(Math.min(got, total)) + ' of ' + mb(total) + ' MB');
    }
    const buf = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    return buf;
  }

  async function loadModel() {
    try {
      const bytes = await fetchModel();
      setStatus('Starting model…');
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      setStatus('Model ready', 'ready');
    } catch (err) {
      console.error(err);
      setStatus('The model could not load: ' + err.message + '. Reload the page to try again.', 'error');
      throw err;
    }
  }
  sessionReady = loadModel();

  // ---------- pre-processing (letterbox, like Ultralytics) ----------
  const prep = document.createElement('canvas');
  prep.width = prep.height = SIZE;
  const pctx = prep.getContext('2d', { willReadFrequently: true });

  function letterbox(img) {
    const w = img.width, h = img.height;
    const r = Math.min(SIZE / w, SIZE / h);
    const nw = Math.round(w * r), nh = Math.round(h * r);
    const left = Math.floor((SIZE - nw) / 2), top = Math.floor((SIZE - nh) / 2);
    pctx.fillStyle = 'rgb(114,114,114)';
    pctx.fillRect(0, 0, SIZE, SIZE);
    pctx.imageSmoothingEnabled = true;
    pctx.imageSmoothingQuality = 'high';
    pctx.drawImage(img, left, top, nw, nh);
    const px = pctx.getImageData(0, 0, SIZE, SIZE).data;
    const area = SIZE * SIZE;
    const data = new Float32Array(3 * area);
    for (let i = 0, p = 0; i < area; i++, p += 4) {
      data[i] = px[p] / 255;
      data[i + area] = px[p + 1] / 255;
      data[i + 2 * area] = px[p + 2] / 255;
    }
    return { tensor: new ort.Tensor('float32', data, [1, 3, SIZE, SIZE]), r, left, top };
  }

  // ---------- post-processing: output is [1, 5, 8400] = cx, cy, w, h, score ----------
  function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1), areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-9);
  }

  function decode(output, lb, imgW, imgH) {
    const d = output.data;
    const n = output.dims[2];
    const cand = [];
    for (let i = 0; i < n; i++) {
      const s = d[4 * n + i];
      if (s < FLOOR) continue;
      const cx = d[i], cy = d[n + i], w = d[2 * n + i], h = d[3 * n + i];
      cand.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score: s });
    }
    cand.sort((a, b) => b.score - a.score);
    // Greedy NMS. Doing it once at the floor and filtering later gives the same
    // result as running NMS at any higher threshold.
    const keep = [];
    for (const c of cand) {
      if (keep.length >= MAX_DET) break;
      if (keep.every((k) => iou(k, c) <= IOU)) keep.push(c);
    }
    const clamp = (v, max) => Math.min(Math.max(v, 0), max);
    return keep.map((k) => ({
      x1: clamp((k.x1 - lb.left) / lb.r, imgW),
      y1: clamp((k.y1 - lb.top) / lb.r, imgH),
      x2: clamp((k.x2 - lb.left) / lb.r, imgW),
      y2: clamp((k.y2 - lb.top) / lb.r, imgH),
      score: k.score,
    }));
  }

  // ---------- drawing ----------
  function visible() {
    const t = Number(conf.value) / 100;
    return allDets.filter((d) => d.score >= t);
  }

  // unit = canvas pixels per CSS pixel, so lines and labels stay readable at any size
  function drawDetections(c, img, dets, scale, highlight, unit) {
    c.drawImage(img, 0, 0, img.width * scale, img.height * scale);
    const side = Math.min(img.width, img.height) * scale / unit;
    const lw = Math.max(1.5, side / 350) * unit;
    const fs = Math.round(Math.min(16, Math.max(11, side / 40)) * unit);
    c.font = '600 ' + fs + 'px Barlow, system-ui, sans-serif';
    c.textBaseline = 'top';
    dets.forEach((d, i) => {
      const x = d.x1 * scale, y = d.y1 * scale;
      const w = (d.x2 - d.x1) * scale, h = (d.y2 - d.y1) * scale;
      const dim = highlight >= 0 && highlight !== i;
      c.globalAlpha = dim ? 0.3 : 1;
      c.lineWidth = highlight === i ? lw * 2 : lw;
      c.strokeStyle = BOX;
      c.strokeRect(x, y, w, h);
      let label = (i + 1) + '  ' + Math.round(d.score * 100) + '%';
      if (c.measureText(label).width + fs * 0.7 > w + fs) label = String(i + 1);  // small box: number only
      const tw = c.measureText(label).width + fs * 0.7;
      const th = fs * 1.35;
      const ty = y - th >= 0 ? y - th : y;
      c.fillStyle = BOX;
      c.fillRect(x - lw / 2, ty, tw, th);
      c.fillStyle = '#0b2a2e';
      c.fillText(label, x - lw / 2 + fs * 0.35, ty + fs * 0.18);
    });
    c.globalAlpha = 1;
  }

  function render() {
    if (!image) return;
    const box = stage.getBoundingClientRect();
    const narrow = window.matchMedia('(max-width: 860px)').matches;
    const maxW = box.width - (narrow ? 32 : 48);
    const maxH = narrow ? window.innerHeight * 0.7 : window.innerHeight - 48;
    const scale = Math.min(maxW / image.width, maxH / image.height, 1);
    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.round(image.width * scale), cssH = Math.round(image.height * scale);
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawDetections(ctx, image, busy ? [] : visible(), scale * dpr, active, dpr);
  }

  function renderList() {
    const dets = visible();
    const n = dets.length;
    countEl.textContent = n === 0 ? 'No corrosion found' : n === 1 ? '1 corroded area' : n + ' corroded areas';
    metaEl.textContent = 'Checked in ' + (lastMs / 1000).toFixed(1) + ' s at ' + conf.value + '% minimum confidence.' +
      (n === 0 && allDets.length ? ' Lower the minimum to see weaker matches.' : '');
    list.innerHTML = '';
    dets.forEach((d, i) => {
      const li = document.createElement('li');
      li.tabIndex = 0;
      const pct = Math.round(d.score * 100);
      li.innerHTML = '<span class="num">' + (i + 1) + '</span>' +
        '<span class="bar"><span style="width:' + pct + '%"></span></span>' +
        '<span class="pct">' + pct + '%</span>';
      li.setAttribute('aria-label', 'Area ' + (i + 1) + ', ' + pct + ' percent confidence');
      const on = () => { active = i; li.classList.add('active'); render(); };
      const off = () => { if (active === i) { active = -1; render(); } li.classList.remove('active'); };
      li.addEventListener('mouseenter', on);
      li.addEventListener('mouseleave', off);
      li.addEventListener('focus', on);
      li.addEventListener('blur', off);
      list.appendChild(li);
    });
    downloadBtn.hidden = false;
    results.hidden = false;
  }

  // ---------- running a photo ----------
  async function analyse(blob) {
    if (busy) return;
    let bmp;
    try {
      bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' });
    } catch (e) {
      setStatus('That file could not be opened as an image. Try a JPG or PNG.', 'error');
      return;
    }
    image = bmp;
    allDets = [];
    active = -1;
    busy = true;
    dropzone.hidden = true;
    viewer.hidden = false;
    scanline.hidden = false;
    chooseBtn.textContent = 'Choose another photo';
    render();
    try {
      if (!session) {
        setStatus('Waiting for the model to finish loading…');
        await sessionReady;
      }
      setStatus('Checking photo…', 'ready');
      await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
      const t0 = performance.now();
      const lb = letterbox(bmp);
      const out = await session.run({ [session.inputNames[0]]: lb.tensor });
      allDets = decode(out[session.outputNames[0]], lb, bmp.width, bmp.height);
      lastMs = performance.now() - t0;
      setStatus('Model ready', 'ready');
    } catch (err) {
      console.error(err);
      if (session) setStatus('Detection failed: ' + err.message, 'error');
    } finally {
      busy = false;
      scanline.hidden = true;
      render();
      if (session) renderList();
    }
  }

  function pickFile(file) {
    if (!file) return;
    if (file.type && !file.type.startsWith('image/')) {
      setStatus('That file is not an image. Choose a JPG or PNG photo.', 'error');
      return;
    }
    document.querySelectorAll('.sample').forEach((b) => b.setAttribute('aria-pressed', 'false'));
    analyse(file);
  }

  // ---------- events ----------
  chooseBtn.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => { pickFile(fileInput.files[0]); fileInput.value = ''; });

  ['dragenter', 'dragover'].forEach((t) => document.addEventListener(t, (e) => {
    e.preventDefault();
    stage.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach((t) => document.addEventListener(t, (e) => {
    e.preventDefault();
    if (t === 'drop' || e.relatedTarget === null) stage.classList.remove('dragging');
  }));
  document.addEventListener('drop', (e) => pickFile(e.dataTransfer && e.dataTransfer.files[0]));
  document.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData ? e.clipboardData.items : [])].find((i) => i.type.startsWith('image/'));
    if (item) pickFile(item.getAsFile());
  });

  document.querySelectorAll('.sample').forEach((btn) => {
    btn.setAttribute('aria-pressed', 'false');
    btn.addEventListener('click', async () => {
      document.querySelectorAll('.sample').forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
      const res = await fetch(btn.dataset.src);
      analyse(await res.blob());
    });
  });

  conf.addEventListener('input', () => {
    confOut.textContent = conf.value + '%';
    active = -1;
    if (image && !busy) { render(); renderList(); }
  });

  downloadBtn.addEventListener('click', () => {
    if (!image) return;
    const out = document.createElement('canvas');
    out.width = image.width;
    out.height = image.height;
    drawDetections(out.getContext('2d'), image, visible(), 1, -1, Math.max(1, Math.min(image.width, image.height) / 700));
    out.toBlob((blob) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'corrosion-marked.jpg';
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }, 'image/jpeg', 0.92);
  });

  new ResizeObserver(() => render()).observe(stage);
})();
