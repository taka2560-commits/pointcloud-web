/**
 * Web版 3D点群エディタ & ノイズ除去ツール メインロジック
 * Three.js + OrbitControls による高速点群描画・操作・編集
 * 設定機能: 移動速度、回転感度、FOV、背景色、グリッド、選択色等のカスタマイズと保存
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// --- グローバル状態 ---
let scene, camera, renderer, controls;
let pointCloud = null;          // 現在のThree.js Pointsオブジェクト
let gridHelper, axesHelper;
let raycaster, mouse;
let focusMarker = null;         // クリック位置の視覚マーカー
let circleTexture = null;       // 円形点描画用のテクスチャ

/**
 * 鮮明な円形点群を描画するための丸テクスチャを動的生成
 */
function getOrCreateCircleTexture() {
  if (circleTexture) return circleTexture;
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.beginPath();
  ctx.arc(32, 32, 28, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  circleTexture = new THREE.CanvasTexture(canvas);
  return circleTexture;
}

// アプリケーション設定 (localStorage永続化)
const defaultSettings = {
  wasdSpeed: 1.0,
  rotateSpeed: 1.0,
  zoomSpeed: 1.0,
  enableDblClickFocus: true,
  bgColor: "#0f1117",
  showGrid: true,
  showAxes: true,
  fov: 45,
  selectionColor: "#ef4444",
};
let appSettings = { ...defaultSettings };

// 点群生データ (Float32Array)
let currentPoints = null;       // [x, y, z, x, y, z, ...]
let currentColors = null;       // [r, g, b, r, g, b, ...] (0.0〜1.0)
let originalPoints = null;      // 読み込み直後の初期点群
let selectedIndices = new Set(); // 選択された点インデックス

// Undo / Redo 履歴スタック
const undoStack = [];
const redoStack = [];

// 選択モード状態
let isSelectMode = false;
let isSelecting = false;
const selectStart = { x: 0, y: 0 };
const selectEnd = { x: 0, y: 0 };

// シーク（高さ範囲）状態
let boundsZ = { min: 0, max: 0 };

// キーボードWASD操作状態
const keysPressed = {};
let targetAnimation = null; // スムーズターゲット移動のアニメーション

// --- 初期化 ---
window.addEventListener("DOMContentLoaded", () => {
  loadSavedSettings();
  initThreeJS();
  applySettingsToThree();
  initEventListeners();
  initSettingsUI();
  // 初回起動時にサンプル点群を自動生成して表示
  loadSamplePointCloud();
});

/**
 * Three.js 3Dシーンとカメラ・OrbitControlsの初期化
 */
function initThreeJS() {
  const container = document.getElementById("canvas-container");
  const width = window.innerWidth;
  const height = window.innerHeight;

  // 1. シーン
  scene = new THREE.Scene();
  scene.background = new THREE.Color(appSettings.bgColor);

  // 2. カメラ
  camera = new THREE.PerspectiveCamera(appSettings.fov, width / height, 0.1, 1000);
  camera.position.set(20, 20, 20);

  // 3. レンダラー
  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  // 4. OrbitControls (カメラ自由操作)
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = true;
  controls.rotateSpeed = appSettings.rotateSpeed;
  controls.zoomSpeed = appSettings.zoomSpeed;
  controls.maxDistance = 500;
  controls.minDistance = 0.2;

  // 5. レイキャスター（クリック判定用）
  raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 0.8;
  mouse = new THREE.Vector2();

  // 6. クリック位置のリングマーカー
  const ringGeo = new THREE.RingGeometry(0.3, 0.45, 32);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, side: THREE.DoubleSide, transparent: true, opacity: 0 });
  focusMarker = new THREE.Mesh(ringGeo, ringMat);
  focusMarker.visible = false;
  scene.add(focusMarker);

  // 7. ガイドグリッド & 座標軸
  gridHelper = new THREE.GridHelper(30, 30, 0x38bdf8, 0x334155);
  gridHelper.visible = appSettings.showGrid;
  scene.add(gridHelper);

  axesHelper = new THREE.AxesHelper(5);
  axesHelper.visible = appSettings.showAxes;
  scene.add(axesHelper);

  // ウィンドウリサイズ監視
  window.addEventListener("resize", onWindowResize);

  // アニメーションループ
  animate();
}

/**
 * メインアニメーションループ (WASD移動とスムーズ移動を含む)
 */
function animate() {
  requestAnimationFrame(animate);

  // 1. WASDキーボード移動の更新
  updateKeyboardNavigation();

  // 2. クリック中心移動のスムーズ補間
  updateTargetAnimation();

  // 3. OrbitControlsの更新
  if (controls && controls.enabled) {
    controls.update();
  }

  renderer.render(scene, camera);
}

/**
 * WASDキーによる画面・カメラの前後左右平行移動
 */
function updateKeyboardNavigation() {
  if (!controls || isSelectMode) return;

  const activeEl = document.activeElement;
  if (activeEl && (activeEl.tagName === "INPUT" || activeEl.tagName === "SELECT" || activeEl.tagName === "TEXTAREA")) {
    return;
  }

  const isW = keysPressed["KeyW"] || keysPressed["ArrowUp"];
  const isS = keysPressed["KeyS"] || keysPressed["ArrowDown"];
  const isA = keysPressed["KeyA"] || keysPressed["ArrowLeft"];
  const isD = keysPressed["KeyD"] || keysPressed["ArrowRight"];
  const isE = keysPressed["KeyE"]; // 上昇
  const isQ = keysPressed["KeyQ"]; // 下降

  if (!isW && !isS && !isA && !isD && !isE && !isQ) return;

  const dist = camera.position.distanceTo(controls.target);
  const baseSpeed = Math.max(0.1, dist * 0.025);
  const speed = baseSpeed * appSettings.wasdSpeed; // 設定倍率を適用

  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);

  const right = new THREE.Vector3();
  right.crossVectors(forward, camera.up).normalize();

  const up = camera.up.clone().normalize();
  const moveDelta = new THREE.Vector3();

  if (isW) moveDelta.addScaledVector(forward, speed);
  if (isS) moveDelta.addScaledVector(forward, -speed);
  if (isD) moveDelta.addScaledVector(right, speed);
  if (isA) moveDelta.addScaledVector(right, -speed);
  if (isE) moveDelta.addScaledVector(up, speed);
  if (isQ) moveDelta.addScaledVector(up, -speed);

  camera.position.add(moveDelta);
  controls.target.add(moveDelta);
}

/**
 * クリックした位置を中心に滑らかにカメラターゲットを移動
 */
function updateTargetAnimation() {
  if (!targetAnimation) return;

  const now = performance.now();
  const progress = Math.min(1.0, (now - targetAnimation.startTime) / targetAnimation.duration);
  const ease = 1 - Math.pow(1 - progress, 3);

  controls.target.lerpVectors(targetAnimation.startTarget, targetAnimation.endTarget, ease);
  camera.position.lerpVectors(targetAnimation.startCam, targetAnimation.endCam, ease);

  if (focusMarker.visible) {
    focusMarker.quaternion.copy(camera.quaternion);
    focusMarker.material.opacity = (1.0 - ease) * 0.9;
  }

  if (progress >= 1.0) {
    targetAnimation = null;
    focusMarker.visible = false;
  }
}

/**
 * 画面上のクリック・ダブルクリックで、その点を中心に移動
 */
function focusOnPoint(clientX, clientY) {
  if (!pointCloud || isSelectMode || !appSettings.enableDblClickFocus) return;

  mouse.x = (clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObject(pointCloud);

  if (intersects.length > 0) {
    const hitPoint = intersects[0].point;
    const offset = camera.position.clone().sub(controls.target);
    const endCamPos = hitPoint.clone().add(offset);

    focusMarker.position.copy(hitPoint);
    focusMarker.quaternion.copy(camera.quaternion);
    focusMarker.material.opacity = 0.9;
    focusMarker.visible = true;

    targetAnimation = {
      startTime: performance.now(),
      duration: 350,
      startTarget: controls.target.clone(),
      endTarget: hitPoint.clone(),
      startCam: camera.position.clone(),
      endCam: endCamPos,
    };

    setStatusMessage(`点 (${hitPoint.x.toFixed(2)}, ${hitPoint.y.toFixed(2)}, ${hitPoint.z.toFixed(2)}) を中心にフォーカス`);
  }
}

function onWindowResize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

/**
 * 点群マテリアルに形状（丸・四角・極小点）とサイズを適用
 */
function applyPointMaterialProperties(material, shape, size) {
  if (!material) return;

  if (shape === "circle") {
    // 丸（円形）: Canvasテクスチャと透過アルファテスト
    material.map = getOrCreateCircleTexture();
    material.alphaTest = 0.5;
    material.transparent = true;
    material.sizeAttenuation = true;
    material.size = size;
  } else if (shape === "square") {
    // 四角（正方形）: 標準GL_POINTS
    material.map = null;
    material.alphaTest = 0.0;
    material.transparent = false;
    material.sizeAttenuation = true;
    material.size = size;
  } else if (shape === "pixel") {
    // 極小点（ピクセルドット）: 距離減衰なしのシャープな極小ドット
    material.map = null;
    material.alphaTest = 0.0;
    material.transparent = false;
    material.sizeAttenuation = false;
    material.size = Math.max(1.0, Math.min(size * 1.5, 4.0));
  }

  material.needsUpdate = true;
}

/**
 * 点群データをThree.jsシーンに設定
 */
function setPointCloud(points, colors = null, fit = true) {
  if (pointCloud) {
    scene.remove(pointCloud);
    pointCloud.geometry.dispose();
    pointCloud.material.dispose();
    pointCloud = null;
  }

  currentPoints = points;
  const n = points.length / 3;

  if (n === 0) {
    updateBadges(0);
    return;
  }

  let minZ = Infinity, maxZ = -Infinity;
  for (let i = 2; i < points.length; i += 3) {
    const z = points[i];
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  boundsZ = { min: minZ, max: maxZ };
  initSeekSliders(minZ, maxZ);

  if (colors && colors.length === points.length) {
    currentColors = colors;
  } else {
    currentColors = generateHeightColors(points, minZ, maxZ);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(currentPoints, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(currentColors.slice(), 3));
  geometry.computeBoundingBox();

  const pointSize = parseFloat(document.getElementById("slider-point-size").value) || 1.2;
  const pointShape = document.getElementById("select-point-shape")?.value || "circle";

  const material = new THREE.PointsMaterial({
    vertexColors: true,
  });
  applyPointMaterialProperties(material, pointShape, pointSize);

  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);

  const box = geometry.boundingBox;
  if (box) {
    const diag = box.min.distanceTo(box.max);
    raycaster.params.Points.threshold = Math.max(0.2, diag * 0.02);
  }

  selectedIndices.clear();
  updateBadges(n);

  if (fit) {
    fitView();
  }
}

function generateHeightColors(points, minZ, maxZ) {
  const colors = new Float32Array(points.length);
  const range = Math.max(1e-4, maxZ - minZ);

  for (let i = 0; i < points.length; i += 3) {
    const t = Math.min(1.0, Math.max(0.0, (points[i + 2] - minZ) / range));
    colors[i] = Math.min(1.0, Math.max(0.0, 1.5 - Math.abs(t * 4.0 - 3.0)));     // R
    colors[i + 1] = Math.min(1.0, Math.max(0.0, 1.5 - Math.abs(t * 4.0 - 2.0))); // G
    colors[i + 2] = Math.min(1.0, Math.max(0.0, 1.5 - Math.abs(t * 4.0 - 1.0))); // B
  }
  return colors;
}

/**
 * 全体表示 (Fit to View)
 */
function fitView() {
  if (!pointCloud) return;

  const box = pointCloud.geometry.boundingBox;
  if (!box) return;

  const center = new THREE.Vector3();
  box.getCenter(center);
  const size = new THREE.Vector3();
  box.getSize(size);

  const maxDim = Math.max(size.x, size.y, size.z, 2.0);
  const fov = camera.fov * (Math.PI / 180);
  const cameraDist = (maxDim / 2) / Math.tan(fov / 2) * 1.5;

  controls.target.copy(center);
  camera.position.set(center.x + cameraDist * 0.7, center.y + cameraDist * 0.7, center.z + cameraDist * 0.7);
  camera.lookAt(center);
  controls.update();

  gridHelper.position.y = box.min.y;
  setStatusMessage("点群全体を中央にフィット表示しました");
}

function setViewPreset(type) {
  if (!controls) return;
  const target = controls.target;
  const dist = camera.position.distanceTo(target) || 20;

  switch (type) {
    case "iso":
      camera.position.set(target.x + dist * 0.6, target.y + dist * 0.6, target.z + dist * 0.6);
      break;
    case "top":
      camera.position.set(target.x, target.y + dist, target.z + 0.001);
      break;
    case "front":
      camera.position.set(target.x, target.y, target.z + dist);
      break;
    case "side":
      camera.position.set(target.x + dist, target.y, target.z);
      break;
  }
  camera.lookAt(target);
  controls.update();
}

/**
 * サンプル点群データ生成
 */
function loadSamplePointCloud() {
  showLoading("サンプル点群を生成中...");

  setTimeout(() => {
    const nBody = 9000;
    const nNoise = 300;
    const total = nBody + nNoise;
    const points = new Float32Array(total * 3);

    for (let i = 0; i < nBody; i++) {
      const u = Math.random() * Math.PI * 2;
      const v = Math.acos(Math.random() * 2 - 1);
      const r = 6.0 + (Math.random() - 0.5) * 0.2;
      points[i * 3] = r * Math.sin(v) * Math.cos(u);
      points[i * 3 + 1] = r * Math.cos(v);
      points[i * 3 + 2] = r * Math.sin(v) * Math.sin(u);
    }

    for (let i = nBody; i < total; i++) {
      points[i * 3] = (Math.random() - 0.5) * 26.0;
      points[i * 3 + 1] = (Math.random() - 0.5) * 26.0;
      points[i * 3 + 2] = (Math.random() - 0.5) * 26.0;
    }

    originalPoints = points.slice();
    undoStack.length = 0;
    redoStack.length = 0;
    updateUndoRedoUI();

    setPointCloud(points);
    hideLoading();
  }, 100);
}

/**
 * 手動ノイズ除去
 */
function deleteSelectedPoints() {
  if (!currentPoints || selectedIndices.size === 0) return;

  undoStack.push({
    points: currentPoints.slice(),
    colors: currentColors.slice(),
  });
  redoStack.length = 0;
  updateUndoRedoUI();

  const total = currentPoints.length / 3;
  const remainCount = total - selectedIndices.size;
  const newPoints = new Float32Array(remainCount * 3);
  const newColors = new Float32Array(remainCount * 3);

  let p = 0;
  for (let i = 0; i < total; i++) {
    if (!selectedIndices.has(i)) {
      newPoints[p * 3] = currentPoints[i * 3];
      newPoints[p * 3 + 1] = currentPoints[i * 3 + 1];
      newPoints[p * 3 + 2] = currentPoints[i * 3 + 2];

      newColors[p * 3] = currentColors[i * 3];
      newColors[p * 3 + 1] = currentColors[i * 3 + 1];
      newColors[p * 3 + 2] = currentColors[i * 3 + 2];
      p++;
    }
  }

  selectedIndices.clear();
  setPointCloud(newPoints, newColors, false);
  setStatusMessage(`${total - remainCount} 点のノイズを削除しました`);
}

function undo() {
  if (undoStack.length === 0) return;
  const prev = undoStack.pop();
  redoStack.push({
    points: currentPoints.slice(),
    colors: currentColors.slice(),
  });
  updateUndoRedoUI();
  selectedIndices.clear();
  setPointCloud(prev.points, prev.colors, false);
  setStatusMessage("元に戻す (Undo) を実行しました");
}

function redo() {
  if (redoStack.length === 0) return;
  const next = redoStack.pop();
  undoStack.push({
    points: currentPoints.slice(),
    colors: currentColors.slice(),
  });
  updateUndoRedoUI();
  selectedIndices.clear();
  setPointCloud(next.points, next.colors, false);
  setStatusMessage("やり直し (Redo) を実行しました");
}

function updateUndoRedoUI() {
  document.getElementById("btn-undo").disabled = undoStack.length === 0;
  document.getElementById("btn-redo").disabled = redoStack.length === 0;
}

/**
 * 矩形選択ボックス処理
 */
function processSelectionBox(rect) {
  if (!currentPoints || !pointCloud) return;

  const total = currentPoints.length / 3;
  const width = window.innerWidth;
  const height = window.innerHeight;

  const posAttr = pointCloud.geometry.attributes.position;
  const colAttr = pointCloud.geometry.attributes.color;
  const p = new THREE.Vector3();

  // ハイライト色 (設定値から取得)
  const selColor = new THREE.Color(appSettings.selectionColor);

  selectedIndices.clear();

  for (let i = 0; i < total; i++) {
    p.set(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i));
    p.project(camera);

    if (p.z > 1.0) continue;

    const screenX = ((p.x + 1) * 0.5) * width;
    const screenY = ((-p.y + 1) * 0.5) * height;

    if (
      screenX >= rect.left &&
      screenX <= rect.right &&
      screenY >= rect.top &&
      screenY <= rect.bottom
    ) {
      selectedIndices.add(i);
      colAttr.setXYZ(i, selColor.r, selColor.g, selColor.b);
    } else {
      colAttr.setXYZ(i, currentColors[i * 3], currentColors[i * 3 + 1], currentColors[i * 3 + 2]);
    }
  }

  colAttr.needsUpdate = true;
  setStatusMessage(`${selectedIndices.size} 点を選択中 (Deleteキーで消去)`);
}

function clearSelection() {
  if (!pointCloud) return;
  selectedIndices.clear();
  const colAttr = pointCloud.geometry.attributes.color;
  for (let i = 0; i < currentColors.length / 3; i++) {
    colAttr.setXYZ(i, currentColors[i * 3], currentColors[i * 3 + 1], currentColors[i * 3 + 2]);
  }
  colAttr.needsUpdate = true;
  setStatusMessage("選択を解除しました");
}

/**
 * 高さ（Z）シークバーフィルタ
 */
function initSeekSliders(minZ, maxZ) {
  const minSlider = document.getElementById("slider-seek-min-z");
  const maxSlider = document.getElementById("slider-seek-max-z");

  minSlider.min = minZ;
  minSlider.max = maxZ;
  minSlider.step = (maxZ - minZ) / 200 || 0.1;
  minSlider.value = minZ;

  maxSlider.min = minZ;
  maxSlider.max = maxZ;
  maxSlider.step = minSlider.step;
  maxSlider.value = maxZ;
}

function applyHeightSeek() {
  if (!pointCloud || !currentPoints) return;

  const minZ = parseFloat(document.getElementById("slider-seek-min-z").value);
  const maxZ = parseFloat(document.getElementById("slider-seek-max-z").value);
  const colAttr = pointCloud.geometry.attributes.color;
  const posAttr = pointCloud.geometry.attributes.position;
  const n = currentPoints.length / 3;

  for (let i = 0; i < n; i++) {
    const z = posAttr.getZ(i);
    if (z >= minZ && z <= maxZ) {
      colAttr.setXYZ(i, currentColors[i * 3], currentColors[i * 3 + 1], currentColors[i * 3 + 2]);
    } else {
      colAttr.setXYZ(i, 0.05, 0.05, 0.08);
    }
  }
  colAttr.needsUpdate = true;
}

/**
 * 統計的外れ値除去 (SOR)
 */
function runSOR() {
  if (!currentPoints) return;
  showLoading("SOR自動ノイズ除去を実行中...");

  setTimeout(() => {
    const k = parseInt(document.getElementById("slider-sor-k").value) || 20;
    const stdRatio = parseFloat(document.getElementById("slider-sor-std").value) || 2.0;
    const total = currentPoints.length / 3;

    const meanDists = new Float32Array(total);
    let sumMean = 0;

    for (let i = 0; i < total; i++) {
      const xi = currentPoints[i * 3];
      const yi = currentPoints[i * 3 + 1];
      const zi = currentPoints[i * 3 + 2];

      let dSum = 0;
      let count = 0;
      const step = Math.max(1, Math.floor(total / 300));
      for (let j = 0; j < total; j += step) {
        if (i === j) continue;
        const dx = xi - currentPoints[j * 3];
        const dy = yi - currentPoints[j * 3 + 1];
        const dz = zi - currentPoints[j * 3 + 2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        dSum += dist;
        count++;
      }
      const avg = dSum / count;
      meanDists[i] = avg;
      sumMean += avg;
    }

    const mean = sumMean / total;
    let varSum = 0;
    for (let i = 0; i < total; i++) {
      varSum += (meanDists[i] - mean) ** 2;
    }
    const std = Math.sqrt(varSum / total);
    const threshold = mean + stdRatio * std;

    selectedIndices.clear();
    for (let i = 0; i < total; i++) {
      if (meanDists[i] > threshold) {
        selectedIndices.add(i);
      }
    }

    hideLoading();
    if (selectedIndices.size > 0) {
      deleteSelectedPoints();
      setStatusMessage(`SORにより ${selectedIndices.size} 点の外れ値を自動除去しました`);
    } else {
      setStatusMessage("外れ値ノイズは検出されませんでした");
    }
  }, 50);
}

/**
 * PLY形式エクスポート
 */
function exportPLY() {
  if (!currentPoints || currentPoints.length === 0) {
    alert("エクスポートする点群データがありません。");
    return;
  }

  const n = currentPoints.length / 3;
  let header = "ply\n";
  header += "format ascii 1.0\n";
  header += `element vertex ${n}\n`;
  header += "property float x\n";
  header += "property float y\n";
  header += "property float z\n";
  header += "property uchar red\n";
  header += "property uchar green\n";
  header += "property uchar blue\n";
  header += "end_header\n";

  let body = "";
  for (let i = 0; i < n; i++) {
    const x = currentPoints[i * 3].toFixed(4);
    const y = currentPoints[i * 3 + 1].toFixed(4);
    const z = currentPoints[i * 3 + 2].toFixed(4);
    const r = Math.round(currentColors[i * 3] * 255);
    const g = Math.round(currentColors[i * 3 + 1] * 255);
    const b = Math.round(currentColors[i * 3 + 2] * 255);
    body += `${x} ${y} ${z} ${r} ${g} ${b}\n`;
  }

  const blob = new Blob([header + body], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "pointcloud_cleaned.ply";
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * テキスト点群パース
 */
function parseTextPointCloud(text) {
  const lines = text.trim().split(/\r?\n/);
  const pts = [];
  const cols = [];

  for (let line of lines) {
    line = line.trim();
    if (!line || line.startsWith("#") || line.startsWith("//")) continue;
    const parts = line.split(/[\s,]+/);
    if (parts.length >= 3) {
      const x = parseFloat(parts[0]);
      const y = parseFloat(parts[1]);
      const z = parseFloat(parts[2]);
      if (isNaN(x) || isNaN(y) || isNaN(z)) continue;
      pts.push(x, y, z);

      if (parts.length >= 6) {
        const r = parseFloat(parts[3]) / (parts[3] > 1 ? 255 : 1);
        const g = parseFloat(parts[4]) / (parts[4] > 1 ? 255 : 1);
        const b = parseFloat(parts[5]) / (parts[5] > 1 ? 255 : 1);
        cols.push(r, g, b);
      }
    }
  }

  return {
    points: new Float32Array(pts),
    colors: cols.length === pts.length ? new Float32Array(cols) : null,
  };
}

/**
 * 設定のロード・保存・反映処理
 */
function loadSavedSettings() {
  try {
    const saved = localStorage.getItem("pointcloud_editor_settings");
    if (saved) {
      appSettings = { ...defaultSettings, ...JSON.parse(saved) };
    }
  } catch (e) {
    console.warn("設定読み込み失敗:", e);
  }
}

function saveSettings() {
  try {
    localStorage.setItem("pointcloud_editor_settings", JSON.stringify(appSettings));
  } catch (e) {
    console.warn("設定保存失敗:", e);
  }
}

function applySettingsToThree() {
  if (scene) {
    scene.background = new THREE.Color(appSettings.bgColor);
  }
  if (controls) {
    controls.rotateSpeed = appSettings.rotateSpeed;
    controls.zoomSpeed = appSettings.zoomSpeed;
  }
  if (gridHelper) {
    gridHelper.visible = appSettings.showGrid;
  }
  if (axesHelper) {
    axesHelper.visible = appSettings.showAxes;
  }
  if (camera) {
    camera.fov = appSettings.fov;
    camera.updateProjectionMatrix();
  }
}

/**
 * 設定画面UIのイベントリスナー初期化
 */
function initSettingsUI() {
  const modal = document.getElementById("settings-modal");
  const openBtn = document.getElementById("btn-open-settings");
  const closeBtn = document.getElementById("btn-close-settings");
  const saveCloseBtn = document.getElementById("btn-save-close-settings");
  const resetBtn = document.getElementById("btn-reset-settings");

  // 開く
  openBtn.addEventListener("click", () => {
    syncSettingsToForm();
    modal.classList.add("open");
  });

  // 閉じる
  const closeModal = () => {
    modal.classList.remove("open");
    saveSettings();
  };
  closeBtn.addEventListener("click", closeModal);
  saveCloseBtn.addEventListener("click", closeModal);

  // モーダル枠外クリックで閉じる
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });

  // モーダルタブ切替
  const tabBtns = document.querySelectorAll(".modal-tab-btn");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // 設定入力イベント (リアルタイム反映)
  const wasdInput = document.getElementById("set-wasd-speed");
  wasdInput.addEventListener("input", (e) => {
    appSettings.wasdSpeed = parseFloat(e.target.value);
    document.getElementById("label-set-wasd-speed").textContent = `${appSettings.wasdSpeed.toFixed(1)}x`;
  });

  const rotateInput = document.getElementById("set-rotate-speed");
  rotateInput.addEventListener("input", (e) => {
    appSettings.rotateSpeed = parseFloat(e.target.value);
    document.getElementById("label-set-rotate-speed").textContent = `${appSettings.rotateSpeed.toFixed(1)}x`;
    if (controls) controls.rotateSpeed = appSettings.rotateSpeed;
  });

  const zoomInput = document.getElementById("set-zoom-speed");
  zoomInput.addEventListener("input", (e) => {
    appSettings.zoomSpeed = parseFloat(e.target.value);
    document.getElementById("label-set-zoom-speed").textContent = `${appSettings.zoomSpeed.toFixed(1)}x`;
    if (controls) controls.zoomSpeed = appSettings.zoomSpeed;
  });

  const dblClickInput = document.getElementById("set-enable-dblclick-focus");
  dblClickInput.addEventListener("change", (e) => {
    appSettings.enableDblClickFocus = e.target.checked;
  });

  // 背景色パレット
  const bgBtns = document.querySelectorAll(".bg-color-btn");
  bgBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      bgBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      appSettings.bgColor = btn.dataset.color;
      if (scene) scene.background = new THREE.Color(appSettings.bgColor);
    });
  });

  // グリッド / 座標軸トグル
  const gridInput = document.getElementById("set-show-grid");
  gridInput.addEventListener("change", (e) => {
    appSettings.showGrid = e.target.checked;
    if (gridHelper) gridHelper.visible = appSettings.showGrid;
  });

  const axesInput = document.getElementById("set-show-axes");
  axesInput.addEventListener("change", (e) => {
    appSettings.showAxes = e.target.checked;
    if (axesHelper) axesHelper.visible = appSettings.showAxes;
  });

  // FOV
  const fovInput = document.getElementById("set-fov");
  fovInput.addEventListener("input", (e) => {
    appSettings.fov = parseInt(e.target.value);
    document.getElementById("label-set-fov").textContent = `${appSettings.fov}°`;
    if (camera) {
      camera.fov = appSettings.fov;
      camera.updateProjectionMatrix();
    }
  });

  // 選択色パレット
  const selBtns = document.querySelectorAll(".sel-color-btn");
  selBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      selBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      appSettings.selectionColor = btn.dataset.color;
    });
  });

  // 初期化ボタン
  resetBtn.addEventListener("click", () => {
    if (confirm("すべての設定を初期値に戻しますか？")) {
      appSettings = { ...defaultSettings };
      applySettingsToThree();
      syncSettingsToForm();
      saveSettings();
      setStatusMessage("設定を初期値に戻しました");
    }
  });
}

function syncSettingsToForm() {
  document.getElementById("set-wasd-speed").value = appSettings.wasdSpeed;
  document.getElementById("label-set-wasd-speed").textContent = `${appSettings.wasdSpeed.toFixed(1)}x`;

  document.getElementById("set-rotate-speed").value = appSettings.rotateSpeed;
  document.getElementById("label-set-rotate-speed").textContent = `${appSettings.rotateSpeed.toFixed(1)}x`;

  document.getElementById("set-zoom-speed").value = appSettings.zoomSpeed;
  document.getElementById("label-set-zoom-speed").textContent = `${appSettings.zoomSpeed.toFixed(1)}x`;

  document.getElementById("set-enable-dblclick-focus").checked = appSettings.enableDblClickFocus;

  // 背景色パレットのアクティブ状態
  document.querySelectorAll(".bg-color-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.color.toLowerCase() === appSettings.bgColor.toLowerCase());
  });

  document.getElementById("set-show-grid").checked = appSettings.showGrid;
  document.getElementById("set-show-axes").checked = appSettings.showAxes;

  document.getElementById("set-fov").value = appSettings.fov;
  document.getElementById("label-set-fov").textContent = `${appSettings.fov}°`;

  // 選択色パレットのアクティブ状態
  document.querySelectorAll(".sel-color-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.color.toLowerCase() === appSettings.selectionColor.toLowerCase());
  });
}

/**
 * UIイベントリスナー登録
 */
function initEventListeners() {
  document.getElementById("btn-fit-view").addEventListener("click", fitView);

  document.getElementById("btn-view-iso").addEventListener("click", () => setViewPreset("iso"));
  document.getElementById("btn-view-top").addEventListener("click", () => setViewPreset("top"));
  document.getElementById("btn-view-front").addEventListener("click", () => setViewPreset("front"));
  document.getElementById("btn-view-side").addEventListener("click", () => setViewPreset("side"));

  document.getElementById("btn-load-sample").addEventListener("click", loadSamplePointCloud);

  // 点サイズスライダー
  const sizeSlider = document.getElementById("slider-point-size");
  sizeSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    document.getElementById("label-point-size").textContent = `${val.toFixed(1)} px`;
    if (pointCloud) {
      const shape = document.getElementById("select-point-shape").value;
      applyPointMaterialProperties(pointCloud.material, shape, val);
    }
  });

  // 点の形状（丸・四角・極小点）セレクトボックス
  const shapeSelect = document.getElementById("select-point-shape");
  shapeSelect.addEventListener("change", (e) => {
    if (pointCloud) {
      const size = parseFloat(sizeSlider.value) || 1.2;
      applyPointMaterialProperties(pointCloud.material, e.target.value, size);
      setStatusMessage(`点の形状を「${shapeSelect.options[shapeSelect.selectedIndex].text}」に変更しました`);
    }
  });

  document.getElementById("select-color-mode").addEventListener("change", (e) => {
    if (!pointCloud || !currentPoints) return;
    const mode = e.target.value;
    const n = currentPoints.length / 3;
    const colAttr = pointCloud.geometry.attributes.color;

    if (mode === "height") {
      currentColors = generateHeightColors(currentPoints, boundsZ.min, boundsZ.max);
    } else if (mode === "solid") {
      currentColors = new Float32Array(currentPoints.length);
      for (let i = 0; i < n; i++) {
        currentColors[i * 3] = 0.2;
        currentColors[i * 3 + 1] = 0.75;
        currentColors[i * 3 + 2] = 0.95;
      }
    }
    colAttr.copyArray(currentColors);
    colAttr.needsUpdate = true;
  });

  const selectToggleBtn = document.getElementById("btn-toggle-select");
  selectToggleBtn.addEventListener("click", () => {
    isSelectMode = !isSelectMode;
    controls.enabled = !isSelectMode;
    if (isSelectMode) {
      selectToggleBtn.textContent = "🔲 矩形選択モード: ON";
      selectToggleBtn.classList.add("btn-active");
      setStatusMessage("画面上を左ドラッグしてノイズ点を囲んでください");
    } else {
      selectToggleBtn.textContent = "🔲 矩形選択モード: OFF";
      selectToggleBtn.classList.remove("btn-active");
      clearSelection();
    }
  });

  const boxElem = document.getElementById("selection-box");

  window.addEventListener("mousedown", (e) => {
    if (!isSelectMode || e.button !== 0) return;
    if (e.target.closest(".sidebar") || e.target.closest(".top-nav") || e.target.closest(".bottom-bar") || e.target.closest(".modal-card")) return;

    isSelecting = true;
    selectStart.x = e.clientX;
    selectStart.y = e.clientY;
    boxElem.style.left = `${selectStart.x}px`;
    boxElem.style.top = `${selectStart.y}px`;
    boxElem.style.width = "0px";
    boxElem.style.height = "0px";
    boxElem.style.display = "block";
  });

  window.addEventListener("mousemove", (e) => {
    if (!isSelecting) return;
    selectEnd.x = e.clientX;
    selectEnd.y = e.clientY;

    const left = Math.min(selectStart.x, selectEnd.x);
    const top = Math.min(selectStart.y, selectEnd.y);
    const width = Math.abs(selectStart.x - selectEnd.x);
    const height = Math.abs(selectStart.y - selectEnd.y);

    boxElem.style.left = `${left}px`;
    boxElem.style.top = `${top}px`;
    boxElem.style.width = `${width}px`;
    boxElem.style.height = `${height}px`;
  });

  window.addEventListener("mouseup", (e) => {
    if (!isSelecting) return;
    isSelecting = false;
    boxElem.style.display = "none";

    const left = Math.min(selectStart.x, selectEnd.x);
    const top = Math.min(selectStart.y, selectEnd.y);
    const right = Math.max(selectStart.x, selectEnd.x);
    const bottom = Math.max(selectStart.y, selectEnd.y);

    if (right - left > 5 && bottom - top > 5) {
      processSelectionBox({ left, top, right, bottom });
    }
  });

  window.addEventListener("dblclick", (e) => {
    if (e.target.closest(".sidebar") || e.target.closest(".top-nav") || e.target.closest(".bottom-bar") || e.target.closest(".modal-card")) return;
    focusOnPoint(e.clientX, e.clientY);
  });

  document.getElementById("btn-delete-selected").addEventListener("click", deleteSelectedPoints);
  document.getElementById("btn-clear-selection").addEventListener("click", clearSelection);

  document.getElementById("btn-undo").addEventListener("click", undo);
  document.getElementById("btn-redo").addEventListener("click", redo);

  window.addEventListener("keydown", (e) => {
    keysPressed[e.code] = true;

    if ((e.key === "f" || e.key === "F") && !isInputFocused()) {
      fitView();
    }
    else if ((e.key === "Delete" || e.key === "Backspace") && !isInputFocused()) {
      deleteSelectedPoints();
    }
    else if (e.ctrlKey && e.key === "z") {
      undo();
    }
    else if (e.ctrlKey && e.key === "y") {
      redo();
    }
  });

  window.addEventListener("keyup", (e) => {
    keysPressed[e.code] = false;
  });

  document.getElementById("slider-sor-k").addEventListener("input", (e) => {
    document.getElementById("label-sor-k").textContent = e.target.value;
  });
  document.getElementById("slider-sor-std").addEventListener("input", (e) => {
    document.getElementById("label-sor-std").textContent = parseFloat(e.target.value).toFixed(1);
  });
  document.getElementById("btn-run-sor").addEventListener("click", runSOR);

  document.getElementById("btn-export-ply").addEventListener("click", exportPLY);

  document.getElementById("slider-seek-min-z").addEventListener("input", applyHeightSeek);
  document.getElementById("slider-seek-max-z").addEventListener("input", applyHeightSeek);
  document.getElementById("btn-reset-seek").addEventListener("click", () => {
    initSeekSliders(boundsZ.min, boundsZ.max);
    applyHeightSeek();
  });

  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");

  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      loadFile(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      loadFile(e.target.files[0]);
    }
  });
}

function isInputFocused() {
  const el = document.activeElement;
  return el && (el.tagName === "INPUT" || el.tagName === "SELECT" || el.tagName === "TEXTAREA");
}

function loadFile(file) {
  showLoading(`ファイル読み込み中: ${file.name}...`);
  const reader = new FileReader();

  reader.onload = (e) => {
    const text = e.target.result;
    const { points, colors } = parseTextPointCloud(text);
    if (points.length > 0) {
      originalPoints = points.slice();
      undoStack.length = 0;
      redoStack.length = 0;
      updateUndoRedoUI();
      setPointCloud(points, colors);
      setStatusMessage(`${file.name} を読み込みました`);
    } else {
      alert("点群データを解析できませんでした。XYZまたはPLY形式をお試しください。");
    }
    hideLoading();
  };

  reader.readAsText(file);
}

function updateBadges(count) {
  document.getElementById("point-count-badge").textContent = `点数: ${count.toLocaleString()} 点`;
}

function setStatusMessage(msg) {
  document.getElementById("status-badge").textContent = msg;
}

function showLoading(msg) {
  const overlay = document.getElementById("loading-overlay");
  document.getElementById("loading-text").textContent = msg;
  overlay.style.display = "flex";
}

function hideLoading() {
  document.getElementById("loading-overlay").style.display = "none";
}
