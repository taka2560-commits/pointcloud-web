/**
 * Web版 3D点群エディタ & ノイズ除去ツール メインロジック
 * Three.js + OrbitControls による高速点群描画・操作・編集
 * 新機能: 正射影(オルソ)⇄透視(パース)切り替え、2点間寸法計測ツール
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// --- グローバル状態 ---
let scene, renderer, controls;
let perspectiveCamera, orthographicCamera, activeCamera;
let isOrthographic = false;     // 現在正射影(オルソ)かどうか

let pointCloud = null;          // 現在のThree.js Pointsオブジェクト
let gridHelper, axesHelper;
let raycaster, mouse;
let focusMarker = null;         // クリック位置の視覚マーカー
let circleTexture = null;       // 円形点描画用のテクスチャ

// 距離計測ツール関連
let isMeasureMode = false;
let measurePoints = [];         // 計測用点 [Vector3, Vector3]
let measureGroup = null;        // 計測用の線・マーカーグループ

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
  const aspect = width / height;

  // 1. シーン
  scene = new THREE.Scene();
  scene.background = new THREE.Color(appSettings.bgColor);

  // 2. カメラ (透視投影 & 正射影の両方を初期化)
  perspectiveCamera = new THREE.PerspectiveCamera(appSettings.fov, aspect, 0.1, 2000);
  perspectiveCamera.position.set(20, 20, 20);

  const frustumSize = 30;
  orthographicCamera = new THREE.OrthographicCamera(
    (-frustumSize * aspect) / 2,
    (frustumSize * aspect) / 2,
    frustumSize / 2,
    -frustumSize / 2,
    0.1,
    2000
  );
  orthographicCamera.position.set(20, 20, 20);

  activeCamera = perspectiveCamera; // 初期は透視投影
  isOrthographic = false;

  // 3. レンダラー
  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  // 4. OrbitControls
  controls = new OrbitControls(activeCamera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = true;
  controls.rotateSpeed = appSettings.rotateSpeed;
  controls.zoomSpeed = appSettings.zoomSpeed;
  controls.maxDistance = 1000;
  controls.minDistance = 0.2;

  // 5. レイキャスター
  raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 0.8;
  mouse = new THREE.Vector2();

  // 6. クリック位置マーカー
  const ringGeo = new THREE.RingGeometry(0.3, 0.45, 32);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, side: THREE.DoubleSide, transparent: true, opacity: 0 });
  focusMarker = new THREE.Mesh(ringGeo, ringMat);
  focusMarker.visible = false;
  scene.add(focusMarker);

  // 7. 計測用オブジェクトグループ
  measureGroup = new THREE.Group();
  scene.add(measureGroup);

  // 8. ガイドグリッド & 座標軸
  gridHelper = new THREE.GridHelper(30, 30, 0x38bdf8, 0x334155);
  gridHelper.visible = appSettings.showGrid;
  scene.add(gridHelper);

  axesHelper = new THREE.AxesHelper(5);
  axesHelper.visible = appSettings.showAxes;
  scene.add(axesHelper);

  window.addEventListener("resize", onWindowResize);
  animate();
}

/**
 * 投影方式の切り替え (透視投影 ⇄ 正射影/オルソ)
 */
function toggleProjectionMode(forceMode = null) {
  const targetMode = forceMode !== null ? forceMode : !isOrthographic;
  if (targetMode === isOrthographic) return;

  const width = window.innerWidth;
  const height = window.innerHeight;
  const aspect = width / height;
  const dist = activeCamera.position.distanceTo(controls.target);

  if (targetMode) {
    // 透視投影 → 正射影 (オルソ)
    isOrthographic = true;
    const fovRad = (perspectiveCamera.fov * Math.PI) / 360;
    const frustumHeight = 2 * dist * Math.tan(fovRad);
    const frustumWidth = frustumHeight * aspect;

    orthographicCamera.left = -frustumWidth / 2;
    orthographicCamera.right = frustumWidth / 2;
    orthographicCamera.top = frustumHeight / 2;
    orthographicCamera.bottom = -frustumHeight / 2;
    orthographicCamera.position.copy(perspectiveCamera.position);
    orthographicCamera.quaternion.copy(perspectiveCamera.quaternion);
    orthographicCamera.updateProjectionMatrix();

    activeCamera = orthographicCamera;
    controls.object = orthographicCamera;

    document.getElementById("btn-toggle-projection").textContent = "📐 投影: 平行投影 (正射影)";
    document.getElementById("btn-toggle-projection").classList.add("btn-active");
    setStatusMessage("正射影（平行投影・オルソ）に切り替えました。歪みのない寸法確認が可能です。");
  } else {
    // 正射影 (オルソ) → 透視投影
    isOrthographic = false;
    perspectiveCamera.position.copy(orthographicCamera.position);
    perspectiveCamera.quaternion.copy(orthographicCamera.quaternion);
    perspectiveCamera.updateProjectionMatrix();

    activeCamera = perspectiveCamera;
    controls.object = perspectiveCamera;

    document.getElementById("btn-toggle-projection").textContent = "📐 投影: 透視投影 (パース)";
    document.getElementById("btn-toggle-projection").classList.remove("btn-active");
    setStatusMessage("透視投影（パースペクティブ）に切り替えました。自然な遠近感表示です。");
  }

  controls.update();
}

/**
 * メインアニメーションループ
 */
function animate() {
  requestAnimationFrame(animate);

  updateKeyboardNavigation();
  updateTargetAnimation();

  if (controls && controls.enabled) {
    controls.update();
  }

  renderer.render(scene, activeCamera);
}

/**
 * WASDキーによる移動
 */
function updateKeyboardNavigation() {
  if (!controls || isSelectMode || isMeasureMode) return;

  const activeEl = document.activeElement;
  if (activeEl && (activeEl.tagName === "INPUT" || activeEl.tagName === "SELECT" || activeEl.tagName === "TEXTAREA")) {
    return;
  }

  const isW = keysPressed["KeyW"] || keysPressed["ArrowUp"];
  const isS = keysPressed["KeyS"] || keysPressed["ArrowDown"];
  const isA = keysPressed["KeyA"] || keysPressed["ArrowLeft"];
  const isD = keysPressed["KeyD"] || keysPressed["ArrowRight"];
  const isE = keysPressed["KeyE"];
  const isQ = keysPressed["KeyQ"];

  if (!isW && !isS && !isA && !isD && !isE && !isQ) return;

  const dist = activeCamera.position.distanceTo(controls.target);
  const baseSpeed = Math.max(0.1, dist * 0.025);
  const speed = baseSpeed * appSettings.wasdSpeed;

  const forward = new THREE.Vector3();
  activeCamera.getWorldDirection(forward);

  const right = new THREE.Vector3();
  right.crossVectors(forward, activeCamera.up).normalize();

  const up = activeCamera.up.clone().normalize();
  const moveDelta = new THREE.Vector3();

  if (isW) moveDelta.addScaledVector(forward, speed);
  if (isS) moveDelta.addScaledVector(forward, -speed);
  if (isD) moveDelta.addScaledVector(right, speed);
  if (isA) moveDelta.addScaledVector(right, -speed);
  if (isE) moveDelta.addScaledVector(up, speed);
  if (isQ) moveDelta.addScaledVector(up, -speed);

  activeCamera.position.add(moveDelta);
  controls.target.add(moveDelta);
}

/**
 * クリック位置へのスムーズ移動
 */
function updateTargetAnimation() {
  if (!targetAnimation) return;

  const now = performance.now();
  const progress = Math.min(1.0, (now - targetAnimation.startTime) / targetAnimation.duration);
  const ease = 1 - Math.pow(1 - progress, 3);

  controls.target.lerpVectors(targetAnimation.startTarget, targetAnimation.endTarget, ease);
  activeCamera.position.lerpVectors(targetAnimation.startCam, targetAnimation.endCam, ease);

  if (focusMarker.visible) {
    focusMarker.quaternion.copy(activeCamera.quaternion);
    focusMarker.material.opacity = (1.0 - ease) * 0.9;
  }

  if (progress >= 1.0) {
    targetAnimation = null;
    focusMarker.visible = false;
  }
}

/**
 * クリックした点を中心にフォーカス
 */
function focusOnPoint(clientX, clientY) {
  if (!pointCloud || isSelectMode || isMeasureMode || !appSettings.enableDblClickFocus) return;

  mouse.x = (clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, activeCamera);
  const intersects = raycaster.intersectObject(pointCloud);

  if (intersects.length > 0) {
    const hitPoint = intersects[0].point;
    const offset = activeCamera.position.clone().sub(controls.target);
    const endCamPos = hitPoint.clone().add(offset);

    focusMarker.position.copy(hitPoint);
    focusMarker.quaternion.copy(activeCamera.quaternion);
    focusMarker.material.opacity = 0.9;
    focusMarker.visible = true;

    targetAnimation = {
      startTime: performance.now(),
      duration: 350,
      startTarget: controls.target.clone(),
      endTarget: hitPoint.clone(),
      startCam: activeCamera.position.clone(),
      endCam: endCamPos,
    };

    setStatusMessage(`点 (${hitPoint.x.toFixed(2)}, ${hitPoint.y.toFixed(2)}, ${hitPoint.z.toFixed(2)}) を中心にフォーカス`);
  }
}

/**
 * 2点間距離計測処理
 */
function handleMeasureClick(clientX, clientY) {
  if (!pointCloud || !isMeasureMode) return;

  mouse.x = (clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, activeCamera);
  const intersects = raycaster.intersectObject(pointCloud);

  if (intersects.length === 0) return;

  const hitPoint = intersects[0].point.clone();

  if (measurePoints.length >= 2) {
    clearMeasure();
  }

  measurePoints.push(hitPoint);

  // マーカー球の作成
  const sphereGeo = new THREE.SphereGeometry(0.18, 16, 16);
  const sphereMat = new THREE.MeshBasicMaterial({ color: 0xfacc15 });
  const marker = new THREE.Mesh(sphereGeo, sphereMat);
  marker.position.copy(hitPoint);
  measureGroup.add(marker);

  if (measurePoints.length === 1) {
    setStatusMessage("【計測中】2点目をクリックしてください");
  } else if (measurePoints.length === 2) {
    // 2点間の直線を描画
    const p1 = measurePoints[0];
    const p2 = measurePoints[1];

    const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
    const lineMat = new THREE.LineBasicMaterial({ color: 0xfacc15, linewidth: 3 });
    const line = new THREE.Line(lineGeo, lineMat);
    measureGroup.add(line);

    // 距離計算
    const dist3d = p1.distanceTo(p2);
    const distXY = Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
    const distZ = Math.abs(p1.z - p2.z);

    document.getElementById("measure-dist-3d").textContent = `${dist3d.toFixed(3)} m`;
    document.getElementById("measure-dist-xy").textContent = `${distXY.toFixed(3)} m`;
    document.getElementById("measure-dist-z").textContent = `${distZ.toFixed(3)} m`;
    document.getElementById("measure-result").style.display = "block";

    setStatusMessage(`計測完了: 直線距離 ${dist3d.toFixed(3)} m (水平: ${distXY.toFixed(3)}m, 高低差: ${distZ.toFixed(3)}m)`);
  }
}

function clearMeasure() {
  measurePoints = [];
  while (measureGroup.children.length > 0) {
    const obj = measureGroup.children[0];
    measureGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) obj.material.dispose();
  }
  document.getElementById("measure-result").style.display = "none";
}

function onWindowResize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const aspect = width / height;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  const dist = activeCamera.position.distanceTo(controls.target);
  const fovRad = (perspectiveCamera.fov * Math.PI) / 360;
  const frustumHeight = 2 * dist * Math.tan(fovRad);
  const frustumWidth = frustumHeight * aspect;

  orthographicCamera.left = -frustumWidth / 2;
  orthographicCamera.right = frustumWidth / 2;
  orthographicCamera.top = frustumHeight / 2;
  orthographicCamera.bottom = -frustumHeight / 2;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(width, height);
}

/**
 * 点群マテリアルに形状（丸・四角・極小点）とサイズを適用
 */
function applyPointMaterialProperties(material, shape, size) {
  if (!material) return;

  if (shape === "circle") {
    material.map = getOrCreateCircleTexture();
    material.alphaTest = 0.5;
    material.transparent = true;
    material.sizeAttenuation = true;
    material.size = size;
  } else if (shape === "square") {
    material.map = null;
    material.alphaTest = 0.0;
    material.transparent = false;
    material.sizeAttenuation = true;
    material.size = size;
  } else if (shape === "pixel") {
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
  clearMeasure();
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
    colors[i] = Math.min(1.0, Math.max(0.0, 1.5 - Math.abs(t * 4.0 - 3.0)));
    colors[i + 1] = Math.min(1.0, Math.max(0.0, 1.5 - Math.abs(t * 4.0 - 2.0)));
    colors[i + 2] = Math.min(1.0, Math.max(0.0, 1.5 - Math.abs(t * 4.0 - 1.0)));
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
  const fov = perspectiveCamera.fov * (Math.PI / 180);
  const cameraDist = (maxDim / 2) / Math.tan(fov / 2) * 1.5;

  controls.target.copy(center);
  activeCamera.position.set(center.x + cameraDist * 0.7, center.y + cameraDist * 0.7, center.z + cameraDist * 0.7);
  activeCamera.lookAt(center);

  if (isOrthographic) {
    const aspect = window.innerWidth / window.innerHeight;
    const frustumHeight = maxDim * 1.5;
    orthographicCamera.left = (-frustumHeight * aspect) / 2;
    orthographicCamera.right = (frustumHeight * aspect) / 2;
    orthographicCamera.top = frustumHeight / 2;
    orthographicCamera.bottom = -frustumHeight / 2;
    orthographicCamera.updateProjectionMatrix();
  }

  controls.update();
  gridHelper.position.y = box.min.y;
  setStatusMessage("点群全体を中央にフィット表示しました");
}

function setViewPreset(type) {
  if (!controls) return;
  const target = controls.target;
  const dist = activeCamera.position.distanceTo(target) || 20;

  switch (type) {
    case "iso":
      activeCamera.position.set(target.x + dist * 0.6, target.y + dist * 0.6, target.z + dist * 0.6);
      break;
    case "top":
      activeCamera.position.set(target.x, target.y + dist, target.z + 0.001);
      break;
    case "front":
      activeCamera.position.set(target.x, target.y, target.z + dist);
      break;
    case "side":
      activeCamera.position.set(target.x + dist, target.y, target.z);
      break;
  }
  activeCamera.lookAt(target);
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
  const selColor = new THREE.Color(appSettings.selectionColor);

  selectedIndices.clear();

  for (let i = 0; i < total; i++) {
    p.set(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i));
    p.project(activeCamera);

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
  if (perspectiveCamera) {
    perspectiveCamera.fov = appSettings.fov;
    perspectiveCamera.updateProjectionMatrix();
  }
}

/**
 * 設定画面UIの初期化
 */
function initSettingsUI() {
  const modal = document.getElementById("settings-modal");
  const openBtn = document.getElementById("btn-open-settings");
  const closeBtn = document.getElementById("btn-close-settings");
  const saveCloseBtn = document.getElementById("btn-save-close-settings");
  const resetBtn = document.getElementById("btn-reset-settings");

  openBtn.addEventListener("click", () => {
    syncSettingsToForm();
    modal.classList.add("open");
  });

  const closeModal = () => {
    modal.classList.remove("open");
    saveSettings();
  };
  closeBtn.addEventListener("click", closeModal);
  saveCloseBtn.addEventListener("click", closeModal);

  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });

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

  document.getElementById("set-wasd-speed").addEventListener("input", (e) => {
    appSettings.wasdSpeed = parseFloat(e.target.value);
    document.getElementById("label-set-wasd-speed").textContent = `${appSettings.wasdSpeed.toFixed(1)}x`;
  });

  document.getElementById("set-rotate-speed").addEventListener("input", (e) => {
    appSettings.rotateSpeed = parseFloat(e.target.value);
    document.getElementById("label-set-rotate-speed").textContent = `${appSettings.rotateSpeed.toFixed(1)}x`;
    if (controls) controls.rotateSpeed = appSettings.rotateSpeed;
  });

  document.getElementById("set-zoom-speed").addEventListener("input", (e) => {
    appSettings.zoomSpeed = parseFloat(e.target.value);
    document.getElementById("label-set-zoom-speed").textContent = `${appSettings.zoomSpeed.toFixed(1)}x`;
    if (controls) controls.zoomSpeed = appSettings.zoomSpeed;
  });

  document.getElementById("set-enable-dblclick-focus").addEventListener("change", (e) => {
    appSettings.enableDblClickFocus = e.target.checked;
  });

  document.querySelectorAll(".bg-color-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".bg-color-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      appSettings.bgColor = btn.dataset.color;
      if (scene) scene.background = new THREE.Color(appSettings.bgColor);
    });
  });

  document.getElementById("set-show-grid").addEventListener("change", (e) => {
    appSettings.showGrid = e.target.checked;
    if (gridHelper) gridHelper.visible = appSettings.showGrid;
  });

  document.getElementById("set-show-axes").addEventListener("change", (e) => {
    appSettings.showAxes = e.target.checked;
    if (axesHelper) axesHelper.visible = appSettings.showAxes;
  });

  document.getElementById("set-fov").addEventListener("input", (e) => {
    appSettings.fov = parseInt(e.target.value);
    document.getElementById("label-set-fov").textContent = `${appSettings.fov}°`;
    if (perspectiveCamera) {
      perspectiveCamera.fov = appSettings.fov;
      perspectiveCamera.updateProjectionMatrix();
    }
  });

  document.querySelectorAll(".sel-color-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".sel-color-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      appSettings.selectionColor = btn.dataset.color;
    });
  });

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

  document.querySelectorAll(".bg-color-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.color.toLowerCase() === appSettings.bgColor.toLowerCase());
  });

  document.getElementById("set-show-grid").checked = appSettings.showGrid;
  document.getElementById("set-show-axes").checked = appSettings.showAxes;

  document.getElementById("set-fov").value = appSettings.fov;
  document.getElementById("label-set-fov").textContent = `${appSettings.fov}°`;

  document.querySelectorAll(".sel-color-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.color.toLowerCase() === appSettings.selectionColor.toLowerCase());
  });
}

/**
 * UIイベントリスナー登録
 */
function initEventListeners() {
  // 全体表示
  document.getElementById("btn-fit-view").addEventListener("click", fitView);

  // 投影切替ボタン
  document.getElementById("btn-toggle-projection").addEventListener("click", () => toggleProjectionMode());

  // 視点プリセット
  document.getElementById("btn-view-iso").addEventListener("click", () => setViewPreset("iso"));
  document.getElementById("btn-view-top").addEventListener("click", () => setViewPreset("top"));
  document.getElementById("btn-view-front").addEventListener("click", () => setViewPreset("front"));
  document.getElementById("btn-view-side").addEventListener("click", () => setViewPreset("side"));

  // 距離計測ツール
  const measureToggleBtn = document.getElementById("btn-toggle-measure");
  measureToggleBtn.addEventListener("click", () => {
    isMeasureMode = !isMeasureMode;
    if (isMeasureMode) {
      if (isSelectMode) document.getElementById("btn-toggle-select").click(); // 選択モードは排他
      measureToggleBtn.textContent = "📏 距離計測: ON";
      measureToggleBtn.classList.add("btn-active");
      clearMeasure();
      setStatusMessage("【計測モード】点群上の1点目をクリックしてください");
    } else {
      measureToggleBtn.textContent = "📏 距離計測: OFF";
      measureToggleBtn.classList.remove("btn-active");
      clearMeasure();
      setStatusMessage("計測モードを終了しました");
    }
  });

  document.getElementById("btn-clear-measure").addEventListener("click", clearMeasure);

  // サンプル生成
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

  // 点の形状セレクトボックス
  const shapeSelect = document.getElementById("select-point-shape");
  shapeSelect.addEventListener("change", (e) => {
    if (pointCloud) {
      const size = parseFloat(sizeSlider.value) || 1.2;
      applyPointMaterialProperties(pointCloud.material, e.target.value, size);
      setStatusMessage(`点の形状を「${shapeSelect.options[shapeSelect.selectedIndex].text}」に変更しました`);
    }
  });

  // 配色モード
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

  // 矩形選択モード切替
  const selectToggleBtn = document.getElementById("btn-toggle-select");
  selectToggleBtn.addEventListener("click", () => {
    isSelectMode = !isSelectMode;
    controls.enabled = !isSelectMode;
    if (isSelectMode) {
      if (isMeasureMode) measureToggleBtn.click(); // 計測モードと排他
      selectToggleBtn.textContent = "🔲 矩形選択モード: ON";
      selectToggleBtn.classList.add("btn-active");
      setStatusMessage("画面上を左ドラッグしてノイズ点を囲んでください");
    } else {
      selectToggleBtn.textContent = "🔲 矩形選択モード: OFF";
      selectToggleBtn.classList.remove("btn-active");
      clearSelection();
    }
  });

  // マウスクリック判定 (距離計測 / 矩形選択)
  const boxElem = document.getElementById("selection-box");

  window.addEventListener("mousedown", (e) => {
    if (e.target.closest(".sidebar") || e.target.closest(".top-nav") || e.target.closest(".bottom-bar") || e.target.closest(".modal-card")) return;

    // 距離計測モード時のクリック
    if (isMeasureMode && e.button === 0) {
      handleMeasureClick(e.clientX, e.clientY);
      return;
    }

    // 矩形選択モード時のドラッグ開始
    if (isSelectMode && e.button === 0) {
      isSelecting = true;
      selectStart.x = e.clientX;
      selectStart.y = e.clientY;
      boxElem.style.left = `${selectStart.x}px`;
      boxElem.style.top = `${selectStart.y}px`;
      boxElem.style.width = "0px";
      boxElem.style.height = "0px";
      boxElem.style.display = "block";
    }
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

  // ダブルクリックで中心移動
  window.addEventListener("dblclick", (e) => {
    if (e.target.closest(".sidebar") || e.target.closest(".top-nav") || e.target.closest(".bottom-bar") || e.target.closest(".modal-card")) return;
    focusOnPoint(e.clientX, e.clientY);
  });

  document.getElementById("btn-delete-selected").addEventListener("click", deleteSelectedPoints);
  document.getElementById("btn-clear-selection").addEventListener("click", clearSelection);

  document.getElementById("btn-undo").addEventListener("click", undo);
  document.getElementById("btn-redo").addEventListener("click", redo);

  // ショートカットキー (P: 投影切替, F: フィット, M: 計測, Delete, Ctrl+Z, Ctrl+Y)
  window.addEventListener("keydown", (e) => {
    keysPressed[e.code] = true;

    if (isInputFocused()) return;

    if (e.key === "p" || e.key === "P" || e.code === "Numpad5") {
      toggleProjectionMode();
    } else if (e.key === "f" || e.key === "F") {
      fitView();
    } else if (e.key === "m" || e.key === "M") {
      document.getElementById("btn-toggle-measure").click();
    } else if (e.key === "Delete" || e.key === "Backspace") {
      deleteSelectedPoints();
    } else if (e.ctrlKey && e.key === "z") {
      undo();
    } else if (e.ctrlKey && e.key === "y") {
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

/**
 * 点群ファイル読み込みエントリーポイント (大容量対応)
 */
async function loadFile(file) {
  const budgetVal = parseInt(document.getElementById("select-point-budget")?.value || "1000000", 10);
  const targetBudget = budgetVal > 0 ? budgetVal : 50000000; // 0の場合は最大5000万点

  const fileSizeMB = file.size / (1024 * 1024);
  const fileSizeGB = file.size / (1024 * 1024 * 1024);

  // 拡張子判定
  const ext = file.name.split(".").pop().toLowerCase();
  if (ext === "e57") {
    alert("【E57バイナリデータについて】\n10GB超のE57生データは、同梱のデスクトップツール (python main.py または cli.py) をお使いいただくと、メモリマッピングとC++エンジンにより最速で直接ノイズ除去が可能です。\n\nWeb版ではXYZ、PLY、PTS、CSV形式の大容量データをストリーミング表示・編集いただけます。");
    return;
  }

  // 25MB以上のファイルは自動的に「大容量ストリーミング・サンプリングモード」で読み込み
  if (file.size > 25 * 1024 * 1024) {
    await loadLargeFileStreaming(file, targetBudget);
  } else {
    await loadSmallFile(file, targetBudget);
  }
}

/**
 * 10GB超の大容量点群用 ストリーミング・チャンクサンプリング読み込み
 * ブラウザのメモリ制限(2GB)を回避し、何ギガバイトでも絶対にクラッシュしない設計
 */
async function loadLargeFileStreaming(file, targetBudget) {
  showLoadingProgress(0, file.size, 0, `大容量データ解析中: ${file.name}`);

  const totalBytes = file.size;
  const chunkSize = 16 * 1024 * 1024; // 16MB単位でチャンク読み込み
  let offset = 0;

  // ファイルサイズからおおよその全点数を推定 (1点あたり約35バイトと仮定)
  const estimatedTotalPoints = Math.max(targetBudget, Math.floor(totalBytes / 35));
  // 読み飛ばしステップ (サンプリング比率)
  const sampleStep = Math.max(1, Math.floor(estimatedTotalPoints / targetBudget));

  const pts = [];
  const cols = [];

  let remainder = "";
  let pointCounter = 0;
  let lastYieldTime = performance.now();

  while (offset < totalBytes) {
    const end = Math.min(offset + chunkSize, totalBytes);
    const blob = file.slice(offset, end);
    const chunkText = await blob.text();

    const fullChunkText = remainder + chunkText;
    const lines = fullChunkText.split(/\r?\n/);

    // 最後の途切れた行を次回チャンクに持ち越す
    remainder = lines.pop() || "";

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line || line.startsWith("#") || line.startsWith("//") || line.startsWith("ply") || line.startsWith("format") || line.startsWith("comment") || line.startsWith("element") || line.startsWith("property") || line.startsWith("end_header")) {
        continue;
      }

      pointCounter++;
      // サンプリングステップに応じて間引き抽出
      if (pointCounter % sampleStep !== 0) continue;

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

    offset = end;

    // UIの描画更新（ブラウザをフリーズさせないよう適宜待機）
    const now = performance.now();
    if (now - lastYieldTime > 60 || offset >= totalBytes) {
      showLoadingProgress(offset, totalBytes, pts.length / 3, `大容量ストリーミング中: ${file.name}`);
      await new Promise((resolve) => setTimeout(resolve, 0));
      lastYieldTime = performance.now();
    }
  }

  hideLoading();

  if (pts.length > 0) {
    const pointsArr = new Float32Array(pts);
    const colorsArr = cols.length === pts.length ? new Float32Array(cols) : null;

    originalPoints = pointsArr.slice();
    undoStack.length = 0;
    redoStack.length = 0;
    updateUndoRedoUI();

    setPointCloud(pointsArr, colorsArr);

    const fileSizeStr = totalBytes > 1024 * 1024 * 1024
      ? `${(totalBytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
      : `${(totalBytes / (1024 * 1024)).toFixed(1)} MB`;

    setStatusMessage(`大容量データ (${fileSizeStr}) から ${(pts.length / 3).toLocaleString()} 点をサンプリング表示しました`);
  } else {
    alert("点群データを解析できませんでした。XYZまたはPLY形式をお試しください。");
  }
}

/**
 * 25MB以下の軽量ファイル用 高速一括読み込み
 */
async function loadSmallFile(file, targetBudget) {
  showLoading(`ファイル読み込み中: ${file.name}...`);
  const text = await file.text();
  const { points, colors } = parseTextPointCloud(text);

  if (points.length > 0) {
    originalPoints = points.slice();
    undoStack.length = 0;
    redoStack.length = 0;
    updateUndoRedoUI();
    setPointCloud(points, colors);
    setStatusMessage(`${file.name} を読み込みました (${(points.length / 3).toLocaleString()} 点)`);
  } else {
    alert("点群データを解析できませんでした。XYZまたはPLY形式をお試しください。");
  }
  hideLoading();
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
  document.getElementById("loading-bar-fill").style.width = "0%";
  document.getElementById("loading-subtext").textContent = "準備中...";
  overlay.style.display = "flex";
}

function showLoadingProgress(loaded, total, currentPointsCount, titleMsg) {
  const overlay = document.getElementById("loading-overlay");
  overlay.style.display = "flex";

  const percent = Math.min(100, Math.floor((loaded / total) * 100));
  document.getElementById("loading-bar-fill").style.width = `${percent}%`;

  const loadedStr = loaded > 1024 * 1024 * 1024
    ? `${(loaded / (1024 * 1024 * 1024)).toFixed(2)} GB`
    : `${(loaded / (1024 * 1024)).toFixed(1)} MB`;

  const totalStr = total > 1024 * 1024 * 1024
    ? `${(total / (1024 * 1024 * 1024)).toFixed(2)} GB`
    : `${(total / (1024 * 1024)).toFixed(1)} MB`;

  document.getElementById("loading-text").textContent = titleMsg;
  document.getElementById("loading-subtext").textContent = `進捗: ${loadedStr} / ${totalStr} (${percent}%) | 抽出点数: ${currentPointsCount.toLocaleString()} 点`;
}

function hideLoading() {
  document.getElementById("loading-overlay").style.display = "none";
}
