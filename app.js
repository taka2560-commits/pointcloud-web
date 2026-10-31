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

// 距離計測ツール & 🧭 2点ピック斜め壁アライン関連
let isMeasureMode = false;
let measurePoints = [];         // 計測用点 [Vector3, Vector3]
let measureGroup = null;        // 計測用の線・マーカーグループ
let isWallPickMode = false;     // 🧭 2点ピック斜め壁アライン待機中フラグ
let wallAlignPoints = [];       // ピックした2点 [Vector3, Vector3]
let wallAlignAngleRad = null;   // 2点ラインの水平角度 (rad)
let wallAlignCenter = null;     // 2点の中点 Vector3
let wallAlignViewType = "horiz"; // "horiz" | "vert" | "ortho_front" | "edge"

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

// 点群生データ & マルチレイヤー管理
let currentPoints = null;       // [x, y, z, x, y, z, ...] (Float64Array / Float32Array)
let rawRgbColors = null;        // ファイル本来のRGB色 [r, g, b, ...] (ない場合はnull)
let baseColors = null;          // 現在の配色モードによる基準色 [r, g, b, ...]
let currentColors = null;       // 描画用カラーバッファ [r, g, b, ...]
let layerMask = null;           // Uint8Array (0: ノイズ, 1: 通常, 2: 地面, 3: 主平面, 4+: カスタムレイヤー)
let showNormalLayer = true;     // 有効レイヤー全体表示フラグ
let showNoiseLayer = false;     // ノイズレイヤー表示フラグ
let layersConfig = {};          // { [layerId]: { id, name, visible, locked, color, useLayerColor } }
const LAYER_PALETTE_HEX = ["#8c6638", "#33d18c", "#a659f2", "#f28c26", "#33d9d9", "#f259b3"];

function initDefaultLayersConfig() {
  layersConfig = {
    1: { id: 1, name: "通常点群", visible: showNormalLayer, locked: false, color: "#38bdf8", useLayerColor: false, isNoise: false },
    0: { id: 0, name: "ノイズ (除去点)", visible: showNoiseLayer, locked: false, color: "#f24040", useLayerColor: true, isNoise: true },
  };
}
initDefaultLayersConfig();

/**
 * 指定されたレイヤーIDがノイズレイヤー（手動削除ID=0 または 自動除去フィルタレイヤー）かどうか判定
 */
function isNoiseLayer(lid) {
  if (Number(lid) === 0) return true;
  const lcfg = layersConfig[lid];
  if (!lcfg) return false;
  return Boolean(lcfg.isNoise || String(lcfg.name || "").startsWith("自動除去:"));
}

let originalPoints = null;      // 読み込み直後の初期点群
let selectedIndices = new Set(); // 選択された点インデックス (currentPoints上のインデックス)
let boundingBoxCache = null;    // 全体バウンディングボックスキャッシュ

// Undo / Redo 履歴スタック (layerMaskとlayersConfigのスナップショットを保持)
const undoStack = [];
const redoStack = [];

// 選択モード状態 (brush / rect / circle / polygon) & ノイズ処理専用UX
let isSelectMode = false;
let isPanMode = false;          // ✋ 左ドラッグ平行移動(パン)専用モード
let spacePanActive = false;     // Spaceキー押下中の一時パンモード
let selectShapeMode = "rect";
let isSelecting = false;
const selectStart = { x: 0, y: 0 };
const selectEnd = { x: 0, y: 0 };
let polygonPoints = [];         // 多角形選択の頂点リスト [{x, y}, ...]
let brushRadius = 28;           // ブラシ半径 (px)
let brushStrokePoints = [];     // ブラシドラッグ中の軌跡 [{x, y}, ...]
let instantDeleteMode = false;  // ⚡ 囲み・なぞり即削除モード
let cleanFocusMode = false;     // 🎯 ノイズ処理専用モード
const mouseCurrent = { x: 0, y: 0 };

// 3軸断面シーク（X / Y / Z範囲）状態
let boundsZ = { min: 0, max: 0 };
let boundsXYZ = {
  x: { min: 0, max: 0 },
  y: { min: 0, max: 0 },
  z: { min: 0, max: 0 },
};
let seekRange = {
  x: { min: -Infinity, max: Infinity },
  y: { min: -Infinity, max: Infinity },
  z: { min: -Infinity, max: Infinity },
};
let activeSeekAxis = "z";

// キーボードWASD操作状態
const keysPressed = {};
let targetAnimation = null; // スムーズターゲット移動のアニメーション
let toastTimer = null;      // トースト通知タイマー

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

/**
 * 🧭 2点ピック斜め壁アラインモードの切替 (Uキー)
 */
function toggleWallPickMode(force = null) {
  isWallPickMode = force !== null ? Boolean(force) : !isWallPickMode;
  const pickBtn = document.getElementById("btn-wall-pick");
  if (pickBtn) pickBtn.classList.toggle("btn-active", isWallPickMode);
  if (isWallPickMode) {
    wallAlignPoints = [];
    if (isMeasureMode) document.getElementById("btn-toggle-measure")?.click();
    document.body.style.cursor = "crosshair";
    setStatusMessage("🧭【2点壁ピックモード(U)】斜めの壁に沿って 1点目 と 2点目 を左クリックしてください");
    showToast("🧭 斜めの壁上の2点をクリックしてください");
  } else {
    document.body.style.cursor = isPanMode ? "grab" : "";
  }
}

function handleWallPickClick(clientX, clientY) {
  if (!pointCloud || !isWallPickMode) return;
  mouse.x = (clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, activeCamera);
  const intersects = raycaster.intersectObject(pointCloud);
  if (intersects.length === 0) return;

  const hitPoint = intersects[0].point.clone();
  wallAlignPoints.push(hitPoint);

  const sphereGeo = new THREE.SphereGeometry(0.18, 16, 16);
  const sphereMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
  const marker = new THREE.Mesh(sphereGeo, sphereMat);
  marker.position.copy(hitPoint);
  measureGroup.add(marker);

  if (wallAlignPoints.length === 1) {
    setStatusMessage("🧭【1/2点ピック済】斜めの壁の2点目をクリックしてください");
    showToast("🧭 1点目を取得！続けて壁の2点目をクリックしてください");
  } else if (wallAlignPoints.length >= 2) {
    const p1 = wallAlignPoints[0];
    const p2 = wallAlignPoints[1];
    const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
    const lineMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, linewidth: 3 });
    measureGroup.add(new THREE.Line(lineGeo, lineMat));

    wallAlignAngleRad = Math.atan2(p2.y - p1.y, p2.x - p1.x);
    wallAlignCenter = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
    toggleWallPickMode(false);
    applyWallAlignedView(wallAlignViewType || "horiz");
  }
}

function applyWallAlignedView(viewType = "horiz") {
  if (wallAlignAngleRad === null || !wallAlignCenter) {
    wallAlignViewType = viewType;
    toggleWallPickMode(true);
    return;
  }
  wallAlignViewType = viewType;
  toggleProjectionMode(true);
  const dist = activeCamera.position.distanceTo(controls.target) || 20;
  controls.target.copy(wallAlignCenter);

  const tx = Math.cos(wallAlignAngleRad);
  const ty = Math.sin(wallAlignAngleRad);
  const nx = -Math.sin(wallAlignAngleRad);
  const ny = Math.cos(wallAlignAngleRad);

  if (viewType === "horiz") {
    activeCamera.position.set(wallAlignCenter.x, wallAlignCenter.y, wallAlignCenter.z + dist);
    activeCamera.up.set(nx, ny, 0).normalize();
  } else if (viewType === "vert") {
    activeCamera.position.set(wallAlignCenter.x, wallAlignCenter.y, wallAlignCenter.z + dist);
    activeCamera.up.set(tx, ty, 0).normalize();
  } else if (viewType === "ortho_front") {
    activeCamera.position.set(wallAlignCenter.x - nx * dist, wallAlignCenter.y - ny * dist, wallAlignCenter.z);
    activeCamera.up.set(0, 0, 1);
  } else if (viewType === "edge") {
    activeCamera.position.set(wallAlignCenter.x - tx * dist, wallAlignCenter.y - ty * dist, wallAlignCenter.z);
    activeCamera.up.set(0, 0, 1);
  }
  activeCamera.lookAt(wallAlignCenter);
  controls.update();
  const deg = ((wallAlignAngleRad * 180) / Math.PI).toFixed(1);
  const labels = {
    horiz: "━ 横表示 (上面・壁を水平配置)",
    vert: "┃ 縦表示 (上面・壁を垂直配置)",
    ortho_front: "🧱 直交正面ビュー",
    edge: "🔪 壁真横・断面エッジ",
  };
  setStatusMessage(`🧭 斜め壁(${deg}°)に合わせて『${labels[viewType] || viewType}』へ切り替えました`);
  showToast(`🧭 壁角 ${deg}°: ${labels[viewType] || viewType}`);
}

function flipWallAlignedView() {
  if (wallAlignAngleRad === null) {
    toggleWallPickMode(true);
    return;
  }
  wallAlignAngleRad += Math.PI;
  applyWallAlignedView(wallAlignViewType || "horiz");
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
 * 点群データをThree.jsシーンに設定 (通常レイヤー・ノイズレイヤーの二層管理対応)
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
    updateBadges(0, 0);
    return;
  }

  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;

  for (let i = 0; i < points.length; i += 3) {
    const x = points[i];
    const y = points[i + 1];
    const z = points[i + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }

  boundsZ = { min: minZ, max: maxZ };
  boundsXYZ = {
    x: { min: minX, max: maxX },
    y: { min: minY, max: maxY },
    z: { min: minZ, max: maxZ },
  };
  seekRange = {
    x: { min: minX, max: maxX },
    y: { min: minY, max: maxY },
    z: { min: minZ, max: maxZ },
  };

  // レイヤーマスクの初期化 (すべて通常レイヤー: 1)
  initDefaultLayersConfig();
  layerMask = new Uint8Array(n);
  layerMask.fill(1);

  if (colors && colors.length === points.length) {
    rawRgbColors = new Float32Array(colors);
  } else {
    rawRgbColors = null;
  }

  rebuildBaseColors();

  const geometry = new THREE.BufferGeometry();
  const posFloat32 = currentPoints instanceof Float32Array ? currentPoints.slice() : new Float32Array(currentPoints);
  currentColors = baseColors.slice();

  geometry.setAttribute("position", new THREE.BufferAttribute(posFloat32, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(currentColors, 3));
  geometry.computeBoundingBox();
  boundingBoxCache = geometry.boundingBox ? geometry.boundingBox.clone() : null;

  const pointSize = parseFloat(document.getElementById("slider-point-size").value) || 1.2;
  const pointShape = document.getElementById("select-point-shape")?.value || "circle";

  const material = new THREE.PointsMaterial({
    vertexColors: true,
  });
  applyPointMaterialProperties(material, pointShape, pointSize);

  pointCloud = new THREE.Points(geometry, material);
  pointCloud.frustumCulled = false;
  scene.add(pointCloud);

  if (boundingBoxCache) {
    const diag = boundingBoxCache.min.distanceTo(boundingBoxCache.max);
    raycaster.params.Points.threshold = Math.max(0.2, diag * 0.02);
  }

  selectedIndices.clear();
  polygonPoints = [];
  renderSelectionOverlay();
  clearMeasure();

  syncSeekSlidersToActiveAxis();
  updatePointCloudDisplay();

  if (fit) {
    fitView();
  }
}

/**
 * 現在の配色モードに応じて baseColors を再計算
 */
function rebuildBaseColors() {
  if (!currentPoints) return;
  const mode = document.getElementById("select-color-mode")?.value || "height";
  const n = currentPoints.length / 3;

  if (mode === "rgb" && rawRgbColors && rawRgbColors.length === currentPoints.length) {
    baseColors = rawRgbColors.slice();
  } else if (mode === "viridis") {
    baseColors = generateViridisColors(currentPoints, boundsZ.min, boundsZ.max);
  } else if (mode === "layer" && layerMask) {
    baseColors = new Float32Array(currentPoints.length);
    const colorCache = {};
    for (const [lid, lcfg] of Object.entries(layersConfig)) {
      colorCache[lid] = new THREE.Color(lcfg.color || "#38bdf8");
    }
    for (let i = 0; i < n; i++) {
      const c = colorCache[layerMask[i]] || new THREE.Color("#38bdf8");
      baseColors[i * 3] = c.r;
      baseColors[i * 3 + 1] = c.g;
      baseColors[i * 3 + 2] = c.b;
    }
  } else if (mode === "solid") {
    baseColors = new Float32Array(currentPoints.length);
    for (let i = 0; i < n; i++) {
      baseColors[i * 3] = 0.2;
      baseColors[i * 3 + 1] = 0.75;
      baseColors[i * 3 + 2] = 0.95;
    }
  } else {
    baseColors = generateHeightColors(currentPoints, boundsZ.min, boundsZ.max);
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
 * Viridis カラーマップによる高さグラデーション生成
 */
function generateViridisColors(points, minZ, maxZ) {
  const colors = new Float32Array(points.length);
  const range = Math.max(1e-4, maxZ - minZ);
  const stops = [
    [0.267, 0.005, 0.329],
    [0.230, 0.322, 0.546],
    [0.128, 0.567, 0.551],
    [0.369, 0.788, 0.383],
    [0.993, 0.906, 0.144],
  ];

  for (let i = 0; i < points.length; i += 3) {
    const t = Math.min(1.0, Math.max(0.0, (points[i + 2] - minZ) / range));
    const scaled = t * (stops.length - 1);
    const idx = Math.min(stops.length - 2, Math.floor(scaled));
    const frac = scaled - idx;
    const c0 = stops[idx];
    const c1 = stops[idx + 1];
    colors[i] = c0[0] + (c1[0] - c0[0]) * frac;
    colors[i + 1] = c0[1] + (c1[1] - c0[1]) * frac;
    colors[i + 2] = c0[2] + (c1[2] - c0[2]) * frac;
  }
  return colors;
}

/**
 * 指定インデックスの点が現在表示対象（レイヤー表示ON かつ 3軸断面内）かどうか判定
 */
function isPointVisible(i) {
  if (!layerMask || !currentPoints) return false;
  const lid = layerMask[i];
  const lcfg = layersConfig[lid];
  const isVisible = lcfg ? lcfg.visible : (isNoiseLayer(lid) ? showNoiseLayer : showNormalLayer);
  if (!isVisible) return false;

  const x = currentPoints[i * 3];
  const y = currentPoints[i * 3 + 1];
  const z = currentPoints[i * 3 + 2];

  const eps = 1e-4;
  if (x < seekRange.x.min - eps || x > seekRange.x.max + eps) return false;
  if (y < seekRange.y.min - eps || y > seekRange.y.max + eps) return false;
  if (z < seekRange.z.min - eps || z > seekRange.z.max + eps) return false;
  return true;
}

/**
 * 指定インデックスの点が選択・編集可能（表示中 かつ 🔒未ロック）か判定
 */
function isPointSelectable(i) {
  if (!isPointVisible(i)) return false;
  const lid = layerMask[i];
  const lcfg = layersConfig[lid];
  return !(lcfg && lcfg.locked);
}

/**
 * 指定インデックスの点が有効レイヤーかつ🔒未ロックか判定
 */
function isPointUnlockedNormal(i) {
  if (!layerMask) return false;
  const lid = layerMask[i];
  if (isNoiseLayer(lid)) return false;
  const lcfg = layersConfig[lid];
  return !(lcfg && lcfg.locked);
}

/**
 * レイヤー状態・選択状態・3軸断面範囲をThree.jsバッファへ高速反映
 */
function updatePointCloudDisplay() {
  if (!pointCloud || !currentPoints || !layerMask || !baseColors) return;

  const n = currentPoints.length / 3;
  const posAttr = pointCloud.geometry.attributes.position;
  const colAttr = pointCloud.geometry.attributes.color;
  const selColor = new THREE.Color(appSettings.selectionColor);
  const mode = document.getElementById("select-color-mode")?.value || "height";

  const layerColorCache = {};
  for (const [lid, lcfg] of Object.entries(layersConfig)) {
    layerColorCache[lid] = new THREE.Color(lcfg.color || "#38bdf8");
  }

  let normalCount = 0;
  let noiseCount = 0;

  for (let i = 0; i < n; i++) {
    const lid = layerMask[i];
    const isNormal = !isNoiseLayer(lid);
    if (isNormal) normalCount++;
    else noiseCount++;

    if (isPointVisible(i)) {
      posAttr.setXYZ(i, currentPoints[i * 3], currentPoints[i * 3 + 1], currentPoints[i * 3 + 2]);
      if (selectedIndices.has(i)) {
        if (isNormal) {
          colAttr.setXYZ(i, selColor.r, selColor.g, selColor.b);
        } else {
          colAttr.setXYZ(i, 1.0, 0.75, 0.1);
        }
      } else if (!isNormal) {
        const c = layerColorCache[lid] || layerColorCache[0] || new THREE.Color(0.95, 0.25, 0.25);
        colAttr.setXYZ(i, c.r, c.g, c.b);
      } else {
        const lcfg = layersConfig[lid];
        if (mode === "layer" || (lcfg && lcfg.useLayerColor)) {
          const c = layerColorCache[lid] || new THREE.Color(0.2, 0.8, 0.5);
          colAttr.setXYZ(i, c.r, c.g, c.b);
        } else {
          colAttr.setXYZ(i, baseColors[i * 3], baseColors[i * 3 + 1], baseColors[i * 3 + 2]);
        }
      }
    } else {
      // 非表示の点はカメラの描画範囲外へ退避
      posAttr.setXYZ(i, 1e10, 1e10, 1e10);
    }
  }

  posAttr.needsUpdate = true;
  colAttr.needsUpdate = true;
  updateBadges(normalCount, noiseCount);
  renderLayersListUI();
  updateFloatingSelectionBar(selectedIndices ? selectedIndices.size : 0);
}

/**
 * マルチレイヤー一覧UIと対象レイヤー選択コンボボックスを更新
 */
function renderLayersListUI() {
  const listEl = document.getElementById("custom-layers-list");
  const selectEl = document.getElementById("select-target-layer");
  if (!listEl || !selectEl) return;

  const counts = {};
  if (layerMask) {
    for (let i = 0; i < layerMask.length; i++) {
      const lid = layerMask[i];
      counts[lid] = (counts[lid] || 0) + 1;
    }
  }

  const prevSelected = selectEl.value;
  selectEl.innerHTML = "";
  listEl.innerHTML = "";

  const sortedIds = Object.keys(layersConfig)
    .map(Number)
    .sort((a, b) => (isNoiseLayer(a) !== isNoiseLayer(b) ? (isNoiseLayer(a) ? 1 : -1) : a - b));

  for (const lid of sortedIds) {
    const lcfg = layersConfig[lid];
    const cnt = counts[lid] || 0;

    // 移動先コンボボックス項目
    const opt = document.createElement("option");
    opt.value = String(lid);
    opt.textContent = `${lcfg.locked ? "🔒" : ""}[ID:${lid}] ${lcfg.name}`;
    selectEl.appendChild(opt);

    // レイヤー一覧行
    const row = document.createElement("div");
    row.style.cssText = "display: flex; align-items: center; gap: 4px; font-size: 11px;";

    const chk = document.createElement("input");
    chk.type = "checkbox";
    chk.checked = lcfg.visible;
    chk.addEventListener("change", () => {
      lcfg.visible = chk.checked;
      if (lid === 0) showNoiseLayer = chk.checked;
      if (lid === 1) showNormalLayer = chk.checked;
      updatePointCloudDisplay();
    });

    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = lcfg.color || "#38bdf8";
    colorInput.style.cssText = "width: 18px; height: 18px; border: none; padding: 0; background: transparent; cursor: pointer;";
    colorInput.title = "レイヤー表示色を変更";
    colorInput.addEventListener("input", () => {
      lcfg.color = colorInput.value;
      lcfg.useLayerColor = true;
      rebuildBaseColors();
      updatePointCloudDisplay();
    });

    const nameSpan = document.createElement("span");
    nameSpan.textContent = lcfg.name;
    nameSpan.style.cssText = "flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;";

    const lockBtn = document.createElement("button");
    lockBtn.className = "btn";
    lockBtn.textContent = lcfg.locked ? "🔒" : "🔓";
    lockBtn.title = "レイヤーのロック(保護)切替";
    lockBtn.style.cssText = `padding: 1px 5px; font-size: 10px; background: ${lcfg.locked ? "#b45309" : "#334155"};`;
    lockBtn.addEventListener("click", () => {
      lcfg.locked = !lcfg.locked;
      if (lcfg.locked && layerMask) {
        for (const idx of Array.from(selectedIndices)) {
          if (layerMask[idx] === lid) selectedIndices.delete(idx);
        }
      }
      updatePointCloudDisplay();
      setStatusMessage(`レイヤー「${lcfg.name}」: ${lcfg.locked ? "🔒ロック(保護 ON)" : "🔓ロック解除"}`);
    });

    const cntSpan = document.createElement("span");
    cntSpan.textContent = `${cnt.toLocaleString()}点`;
    cntSpan.style.cssText = `min-width: 52px; text-align: right; color: ${isNoiseLayer(lid) ? "#f87171" : "#cbd5e1"};`;

    row.appendChild(chk);
    row.appendChild(colorInput);
    row.appendChild(nameSpan);
    row.appendChild(lockBtn);
    row.appendChild(cntSpan);
    listEl.appendChild(row);
  }

  if (prevSelected && layersConfig[Number(prevSelected)]) {
    selectEl.value = prevSelected;
  }
}

/**
 * 全体表示 (Fit to View)
 */
function fitView() {
  if (!pointCloud) return;

  const box = boundingBoxCache || pointCloud.geometry.boundingBox;
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
 * Undo用に現在のレイヤーマスクとレイヤー設定を履歴スタックへ保存
 */
function pushUndoState() {
  if (!layerMask) return;
  undoStack.push({
    mask: layerMask.slice(),
    layers: JSON.parse(JSON.stringify(layersConfig)),
  });
  if (undoStack.length > 30) undoStack.shift();
  redoStack.length = 0;
  updateUndoRedoUI();
}

/**
 * 手動ノイズ除去 (選択された未ロック有効点をノイズレイヤーへ移動)
 */
function deleteSelectedPoints() {
  if (!currentPoints || !layerMask || selectedIndices.size === 0) return;

  let targetCount = 0;
  for (const idx of selectedIndices) {
    if (isPointUnlockedNormal(idx)) targetCount++;
  }

  if (targetCount === 0) {
    setStatusMessage("選択範囲内に削除可能な未ロック有効点がありません");
    return;
  }

  pushUndoState();

  for (const idx of selectedIndices) {
    if (isPointUnlockedNormal(idx)) {
      layerMask[idx] = 0;
    }
  }

  selectedIndices.clear();
  updatePointCloudDisplay();
  setStatusMessage(`${targetCount.toLocaleString()} 点をノイズレイヤーへ移動しました`);
  showToast(`🗑 ${targetCount.toLocaleString()} 点をノイズレイヤーへ移動しました`);
}

/**
 * ① カスタムレイヤーを追加
 */
function addCustomLayer() {
  const inputEl = document.getElementById("input-custom-layer-name");
  let name = inputEl ? inputEl.value.trim() : "";
  let newId = 4;
  while (layersConfig[newId]) newId++;
  if (!name) name = `カスタムレイヤー ${newId - 3}`;
  const color = LAYER_PALETTE_HEX[(newId - 2) % LAYER_PALETTE_HEX.length];
  layersConfig[newId] = {
    id: newId,
    name,
    visible: true,
    locked: false,
    color,
    useLayerColor: true,
  };
  if (inputEl) inputEl.value = "";
  renderLayersListUI();
  const selectEl = document.getElementById("select-target-layer");
  if (selectEl) selectEl.value = String(newId);
  setStatusMessage(`新規レイヤー「${name}」を追加しました`);
  showToast(`✨ レイヤー「${name}」を追加しました`);
}

/**
 * ① 選択中の点を指定レイヤーへ移動
 */
function moveSelectedToTargetLayer() {
  if (!currentPoints || !layerMask || selectedIndices.size === 0) {
    setStatusMessage("移動したい点を先に選択してください");
    return;
  }
  const selectEl = document.getElementById("select-target-layer");
  const targetId = selectEl ? Number(selectEl.value) : 1;
  const targetCfg = layersConfig[targetId];
  if (!targetCfg) return;
  if (targetCfg.locked) {
    setStatusMessage(`移動先レイヤー「${targetCfg.name}」がロックされているため移動できません`);
    return;
  }

  pushUndoState();
  let moved = 0;
  for (const idx of selectedIndices) {
    const curCfg = layersConfig[layerMask[idx]];
    if (curCfg && curCfg.locked) continue;
    layerMask[idx] = targetId;
    moved++;
  }
  selectedIndices.clear();
  rebuildBaseColors();
  updatePointCloudDisplay();
  setStatusMessage(`${moved.toLocaleString()} 点をレイヤー「${targetCfg.name}」へ移動しました`);
  showToast(`📁 ${moved.toLocaleString()} 点を「${targetCfg.name}」へ移動しました`);
}

/**
 * ⑥ 局所最低標高グリッド法による地面の自動抽出 → 地面レイヤー(ID:2)へ移動
 */
function autoExtractGround() {
  if (!currentPoints || !layerMask) return;
  const n = currentPoints.length / 3;
  const gridSize = 1.5;
  const heightThresh = 0.35;
  const minZMap = new Map();

  for (let i = 0; i < n; i++) {
    if (!isPointUnlockedNormal(i)) continue;
    const gx = Math.floor((currentPoints[i * 3] - boundsXYZ.x.min) / gridSize);
    const gz = Math.floor((currentPoints[i * 3 + 2] - boundsXYZ.z.min) / gridSize);
    const key = `${gx}_${gz}`;
    const y = currentPoints[i * 3 + 1];
    const prev = minZMap.get(key);
    if (prev === undefined || y < prev) minZMap.set(key, y);
  }

  const groundHits = [];
  for (let i = 0; i < n; i++) {
    if (!isPointUnlockedNormal(i)) continue;
    const gx = Math.floor((currentPoints[i * 3] - boundsXYZ.x.min) / gridSize);
    const gz = Math.floor((currentPoints[i * 3 + 2] - boundsXYZ.z.min) / gridSize);
    const minY = minZMap.get(`${gx}_${gz}`);
    if (minY !== undefined && currentPoints[i * 3 + 1] - minY <= heightThresh) {
      groundHits.push(i);
    }
  }

  if (groundHits.length === 0) {
    setStatusMessage("地面として抽出可能な点が見つかりませんでした");
    return;
  }

  pushUndoState();
  if (!layersConfig[2]) {
    layersConfig[2] = { id: 2, name: "地面 (自動抽出)", visible: true, locked: false, color: "#8c6638", useLayerColor: true };
  }
  for (const idx of groundHits) {
    layerMask[idx] = 2;
  }
  rebuildBaseColors();
  updatePointCloudDisplay();
  setStatusMessage(`地面 ${groundHits.length.toLocaleString()} 点を「地面 (自動抽出)」レイヤーへ分離しました`);
  showToast(`🏞️ 地面 ${groundHits.length.toLocaleString()} 点を自動抽出しました`);
}

/**
 * ⑥ RANSAC法による主平面(壁/床)の自動検出 → 平面レイヤー(ID:3)へ移動
 */
function autoExtractPlane() {
  if (!currentPoints || !layerMask) return;
  const n = currentPoints.length / 3;
  const cand = [];
  for (let i = 0; i < n; i++) {
    if (isPointUnlockedNormal(i)) cand.push(i);
  }
  if (cand.length < 3) {
    setStatusMessage("平面検出に必要な未ロック有効点が不足しています");
    return;
  }

  const sampleStep = Math.max(1, Math.floor(cand.length / 1500));
  let bestNormal = [0, 1, 0];
  let bestD = 0;
  let bestCount = -1;
  const distThresh = 0.15;

  for (let iter = 0; iter < 120; iter++) {
    const i1 = cand[Math.floor(Math.random() * cand.length)];
    const i2 = cand[Math.floor(Math.random() * cand.length)];
    const i3 = cand[Math.floor(Math.random() * cand.length)];
    if (i1 === i2 || i2 === i3 || i1 === i3) continue;

    const ux = currentPoints[i2 * 3] - currentPoints[i1 * 3];
    const uy = currentPoints[i2 * 3 + 1] - currentPoints[i1 * 3 + 1];
    const uz = currentPoints[i2 * 3 + 2] - currentPoints[i1 * 3 + 2];
    const vx = currentPoints[i3 * 3] - currentPoints[i1 * 3];
    const vy = currentPoints[i3 * 3 + 1] - currentPoints[i1 * 3 + 1];
    const vz = currentPoints[i3 * 3 + 2] - currentPoints[i1 * 3 + 2];

    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz);
    if (len < 1e-6) continue;
    nx /= len; ny /= len; nz /= len;
    const d = -(nx * currentPoints[i1 * 3] + ny * currentPoints[i1 * 3 + 1] + nz * currentPoints[i1 * 3 + 2]);

    let count = 0;
    for (let k = 0; k < cand.length; k += sampleStep) {
      const idx = cand[k];
      const dist = Math.abs(nx * currentPoints[idx * 3] + ny * currentPoints[idx * 3 + 1] + nz * currentPoints[idx * 3 + 2] + d);
      if (dist <= distThresh) count++;
    }
    if (count > bestCount) {
      bestCount = count;
      bestNormal = [nx, ny, nz];
      bestD = d;
    }
  }

  const planeHits = [];
  for (const idx of cand) {
    const dist = Math.abs(
      bestNormal[0] * currentPoints[idx * 3] +
      bestNormal[1] * currentPoints[idx * 3 + 1] +
      bestNormal[2] * currentPoints[idx * 3 + 2] +
      bestD
    );
    if (dist <= distThresh) planeHits.push(idx);
  }

  if (planeHits.length === 0) {
    setStatusMessage("主平面が検出されませんでした");
    return;
  }

  pushUndoState();
  if (!layersConfig[3]) {
    layersConfig[3] = { id: 3, name: "主平面 (壁/床)", visible: true, locked: false, color: "#33d18c", useLayerColor: true };
  }
  for (const idx of planeHits) {
    layerMask[idx] = 3;
  }
  rebuildBaseColors();
  updatePointCloudDisplay();
  setStatusMessage(`主平面 ${planeHits.length.toLocaleString()} 点を「主平面 (壁/床)」レイヤーへ分離しました`);
  showToast(`🧱 主平面 ${planeHits.length.toLocaleString()} 点を自動検出しました`);
}

/**
 * ⑤ 作業途中のレイヤー状態をJSONファイルとして保存・再開
 */
function saveLayerProjectJSON() {
  if (!layerMask) return;
  const payload = {
    pointCount: layerMask.length,
    layersConfig,
    layerMask: Array.from(layerMask),
  };
  const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "pointcloud_layers_project.json";
  a.click();
  URL.revokeObjectURL(url);
  setStatusMessage("レイヤー状態プロジェクト (.json) を保存しました");
  showToast("💾 レイヤー状態JSONを保存しました");
}

async function loadLayerProjectJSON(file) {
  if (!layerMask) return;
  try {
    const text = await file.text();
    const data = JSON.parse(text);
    if (!Array.isArray(data.layerMask) || data.layerMask.length !== layerMask.length) {
      alert(`レイヤー状態の点数(${data.layerMask?.length})が現在の点群点数(${layerMask.length})と一致しません。`);
      return;
    }
    pushUndoState();
    if (data.layersConfig) layersConfig = data.layersConfig;
    layerMask = new Uint8Array(data.layerMask);
    rebuildBaseColors();
    updatePointCloudDisplay();
    setStatusMessage("レイヤー状態プロジェクトを復元しました");
    showToast("📂 レイヤー状態を復元しました");
  } catch (err) {
    alert(`レイヤー状態ファイルの読み込みに失敗しました: ${err}`);
  }
}

/**
 * 選択したノイズ点のみを通常レイヤーへピンポイント復元
 */
function restoreSelectedNoise() {
  if (!currentPoints || !layerMask || selectedIndices.size === 0) {
    setStatusMessage("復元するノイズ点を先に選択してください（ノイズレイヤー表示ONで選択可能）");
    return;
  }

  let restoreCount = 0;
  for (const idx of selectedIndices) {
    if (isNoiseLayer(layerMask[idx])) restoreCount++;
  }

  if (restoreCount === 0) {
    setStatusMessage("選択された点の中にノイズレイヤーの点がありません");
    return;
  }

  pushUndoState();

  for (const idx of selectedIndices) {
    if (isNoiseLayer(layerMask[idx])) {
      layerMask[idx] = 1;
    }
  }

  selectedIndices.clear();
  updatePointCloudDisplay();
  setStatusMessage(`選択した ${restoreCount.toLocaleString()} 点のノイズを通常レイヤーへ復元しました`);
  showToast(`♻ 選択ノイズ ${restoreCount.toLocaleString()} 点を通常レイヤーへ復元しました`);
}

/**
 * すべてのノイズ点を通常レイヤーへ一括復元
 */
function restoreAllNoise() {
  if (!currentPoints || !layerMask) return;

  let noiseCount = 0;
  for (let i = 0; i < layerMask.length; i++) {
    if (isNoiseLayer(layerMask[i])) noiseCount++;
  }

  if (noiseCount === 0) {
    setStatusMessage("ノイズレイヤーに点が存在しません");
    return;
  }

  pushUndoState();
  for (let i = 0; i < layerMask.length; i++) {
    if (isNoiseLayer(layerMask[i])) layerMask[i] = 1;
  }
  selectedIndices.clear();
  updatePointCloudDisplay();
  setStatusMessage(`すべてのノイズ (${noiseCount.toLocaleString()} 点) を通常レイヤーへ復元しました`);
  showToast(`♻ 全ノイズ ${noiseCount.toLocaleString()} 点を通常レイヤーへ復元しました`);
}

function undo() {
  if (undoStack.length === 0 || !layerMask) return;
  redoStack.push({
    mask: layerMask.slice(),
    layers: JSON.parse(JSON.stringify(layersConfig)),
  });
  const snap = undoStack.pop();
  layerMask = snap.mask;
  if (snap.layers) layersConfig = snap.layers;
  updateUndoRedoUI();
  selectedIndices.clear();
  rebuildBaseColors();
  updatePointCloudDisplay();
  setStatusMessage("元に戻す (Undo) を実行しました");
  showToast("↩ 操作を元に戻しました (Undo)");
}

function redo() {
  if (redoStack.length === 0 || !layerMask) return;
  undoStack.push({
    mask: layerMask.slice(),
    layers: JSON.parse(JSON.stringify(layersConfig)),
  });
  const snap = redoStack.pop();
  layerMask = snap.mask;
  if (snap.layers) layersConfig = snap.layers;
  updateUndoRedoUI();
  selectedIndices.clear();
  rebuildBaseColors();
  updatePointCloudDisplay();
  setStatusMessage("やり直し (Redo) を実行しました");
  showToast("↪ 操作をやり直しました (Redo)");
}

function updateUndoRedoUI() {
  const canUndo = undoStack.length === 0;
  const canRedo = redoStack.length === 0;
  const btnUndo = document.getElementById("btn-undo");
  const btnRedo = document.getElementById("btn-redo");
  const btnAutoUndo = document.getElementById("btn-auto-undo");
  const btnAutoRedo = document.getElementById("btn-auto-redo");
  if (btnUndo) btnUndo.disabled = canUndo;
  if (btnRedo) btnRedo.disabled = canRedo;
  if (btnAutoUndo) btnAutoUndo.disabled = canUndo;
  if (btnAutoRedo) btnAutoRedo.disabled = canRedo;
}

/**
 * 修飾キー (Shift追加 / Ctrl除外) を考慮して選択インデックス集合を更新 (即時削除モード時は即消去)
 */
function applySelectionHits(hitIndices, shiftKey = false, ctrlKey = false) {
  if (instantDeleteMode && !ctrlKey) {
    if (hitIndices.length === 0) {
      setStatusMessage("対象範囲内に点はありませんでした");
      return;
    }
    pushUndoState();
    for (const idx of hitIndices) {
      layerMask[idx] = 0;
    }
    selectedIndices.clear();
    updatePointCloudDisplay();
    updateUndoRedoUI();
    setStatusMessage(`⚡ ${hitIndices.length.toLocaleString()} 点を即座にノイズ消去しました (Ctrl+Zで取消可)`);
    showToast(`⚡ ${hitIndices.length.toLocaleString()} 点を即消去しました`);
    return;
  }

  if (shiftKey) {
    for (const idx of hitIndices) selectedIndices.add(idx);
  } else if (ctrlKey) {
    for (const idx of hitIndices) selectedIndices.delete(idx);
  } else {
    selectedIndices = new Set(hitIndices);
  }
  updatePointCloudDisplay();
  setStatusMessage(`${selectedIndices.size.toLocaleString()} 点を選択中 (Deleteでノイズ移動 / 選択点を別レイヤーへ移動可)`);
}

/**
 * ブラシなぞり軌跡 (カプセル連続線分) による選択処理
 */
function processSelectionBrush(strokePts, radius, shiftKey = false, ctrlKey = false) {
  if (!currentPoints || !pointCloud || !strokePts || strokePts.length === 0) return;

  const total = currentPoints.length / 3;
  const width = window.innerWidth;
  const height = window.innerHeight;
  const r2 = radius * radius;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const pt of strokePts) {
    if (pt.x < minX) minX = pt.x;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.y > maxY) maxY = pt.y;
  }
  minX -= radius;
  maxX += radius;
  minY -= radius;
  maxY += radius;

  const p = new THREE.Vector3();
  const hits = [];

  for (let i = 0; i < total; i++) {
    if (!isPointSelectable(i)) continue;
    p.set(currentPoints[i * 3], currentPoints[i * 3 + 1], currentPoints[i * 3 + 2]);
    p.project(activeCamera);
    if (p.z > 1.0 || p.z < -1.0) continue;

    const sx = ((p.x + 1) * 0.5) * width;
    const sy = ((-p.y + 1) * 0.5) * height;
    if (sx < minX || sx > maxX || sy < minY || sy > maxY) continue;

    let hit = false;
    if (strokePts.length === 1) {
      const dx = sx - strokePts[0].x;
      const dy = sy - strokePts[0].y;
      hit = (dx * dx + dy * dy <= r2);
    } else {
      for (let s = 0; s < strokePts.length - 1; s++) {
        const x0 = strokePts[s].x, y0 = strokePts[s].y;
        const x1 = strokePts[s + 1].x, y1 = strokePts[s + 1].y;
        const dxSeg = x1 - x0, dySeg = y1 - y0;
        const lenSq = dxSeg * dxSeg + dySeg * dySeg;
        let distSq;
        if (lenSq < 1e-4) {
          distSq = (sx - x0) ** 2 + (sy - y0) ** 2;
        } else {
          const t = Math.max(0, Math.min(1, ((sx - x0) * dxSeg + (sy - y0) * dySeg) / lenSq));
          const px = x0 + t * dxSeg;
          const py = y0 + t * dySeg;
          distSq = (sx - px) ** 2 + (sy - py) ** 2;
        }
        if (distSq <= r2) {
          hit = true;
          break;
        }
      }
    }
    if (hit) hits.push(i);
  }

  applySelectionHits(hits, shiftKey, ctrlKey);
}

/**
 * 矩形選択ボックス処理
 */
function processSelectionBox(rect, shiftKey = false, ctrlKey = false) {
  if (!currentPoints || !pointCloud) return;

  const total = currentPoints.length / 3;
  const width = window.innerWidth;
  const height = window.innerHeight;
  const p = new THREE.Vector3();
  const hits = [];

  for (let i = 0; i < total; i++) {
    if (!isPointSelectable(i)) continue;
    p.set(currentPoints[i * 3], currentPoints[i * 3 + 1], currentPoints[i * 3 + 2]);
    p.project(activeCamera);
    if (p.z > 1.0 || p.z < -1.0) continue;

    const screenX = ((p.x + 1) * 0.5) * width;
    const screenY = ((-p.y + 1) * 0.5) * height;

    if (
      screenX >= rect.left &&
      screenX <= rect.right &&
      screenY >= rect.top &&
      screenY <= rect.bottom
    ) {
      hits.push(i);
    }
  }

  applySelectionHits(hits, shiftKey, ctrlKey);
}

/**
 * 円形選択処理
 */
function processSelectionCircle(cx, cy, radius, shiftKey = false, ctrlKey = false) {
  if (!currentPoints || !pointCloud) return;

  const total = currentPoints.length / 3;
  const width = window.innerWidth;
  const height = window.innerHeight;
  const r2 = radius * radius;
  const p = new THREE.Vector3();
  const hits = [];

  for (let i = 0; i < total; i++) {
    if (!isPointSelectable(i)) continue;
    p.set(currentPoints[i * 3], currentPoints[i * 3 + 1], currentPoints[i * 3 + 2]);
    p.project(activeCamera);
    if (p.z > 1.0 || p.z < -1.0) continue;

    const screenX = ((p.x + 1) * 0.5) * width;
    const screenY = ((-p.y + 1) * 0.5) * height;
    const dx = screenX - cx;
    const dy = screenY - cy;
    if (dx * dx + dy * dy <= r2) {
      hits.push(i);
    }
  }

  applySelectionHits(hits, shiftKey, ctrlKey);
}

/**
 * 多角形内外判定 (Ray-Casting Even-Odd Rule)
 */
function isPointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y;
    const xj = poly[j].x, yj = poly[j].y;
    const intersect = ((yi > y) !== (yj > y)) &&
      (x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

/**
 * 多角形選択確定処理
 */
function confirmPolygonSelection(shiftKey = false, ctrlKey = false) {
  if (!currentPoints || !pointCloud || polygonPoints.length < 3) return;

  const total = currentPoints.length / 3;
  const width = window.innerWidth;
  const height = window.innerHeight;
  const p = new THREE.Vector3();
  const hits = [];

  for (let i = 0; i < total; i++) {
    if (!isPointSelectable(i)) continue;
    p.set(currentPoints[i * 3], currentPoints[i * 3 + 1], currentPoints[i * 3 + 2]);
    p.project(activeCamera);
    if (p.z > 1.0 || p.z < -1.0) continue;

    const screenX = ((p.x + 1) * 0.5) * width;
    const screenY = ((-p.y + 1) * 0.5) * height;
    if (isPointInPolygon(screenX, screenY, polygonPoints)) {
      hits.push(i);
    }
  }

  polygonPoints = [];
  const confirmBtn = document.getElementById("btn-confirm-polygon");
  if (confirmBtn) confirmBtn.disabled = true;
  renderSelectionOverlay();
  applySelectionHits(hits, shiftKey, ctrlKey);
}

/**
 * ブラシ・円形・多角形選択用のSVGオーバーレイを描画
 */
function renderSelectionOverlay() {
  const svg = document.getElementById("selection-svg");
  if (!svg) return;
  svg.innerHTML = "";

  if (!isSelectMode) return;

  const strokeCol = instantDeleteMode ? "#f87171" : "#38bdf8";
  const fillCol = instantDeleteMode ? "rgba(239, 68, 68, 0.22)" : "rgba(56, 189, 248, 0.15)";

  if (selectShapeMode === "brush") {
    if (isSelecting && brushStrokePoints.length > 0) {
      const ptsAttr = brushStrokePoints.map((pt) => `${pt.x},${pt.y}`).join(" ");
      const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      polyline.setAttribute("points", ptsAttr);
      polyline.setAttribute("fill", "none");
      polyline.setAttribute("stroke", instantDeleteMode ? "rgba(239, 68, 68, 0.38)" : "rgba(14, 165, 233, 0.35)");
      polyline.setAttribute("stroke-width", String(brushRadius * 2));
      polyline.setAttribute("stroke-linecap", "round");
      polyline.setAttribute("stroke-linejoin", "round");
      svg.appendChild(polyline);
    }
    if (mouseCurrent.x > 0 || mouseCurrent.y > 0) {
      const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      ring.setAttribute("cx", mouseCurrent.x);
      ring.setAttribute("cy", mouseCurrent.y);
      ring.setAttribute("r", brushRadius);
      ring.setAttribute("fill", fillCol);
      ring.setAttribute("stroke", strokeCol);
      ring.setAttribute("stroke-width", "2");
      svg.appendChild(ring);
    }
  } else if (selectShapeMode === "circle" && isSelecting) {
    const cx = selectStart.x;
    const cy = selectStart.y;
    const r = Math.hypot(selectEnd.x - cx, selectEnd.y - cy);
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", cx);
    circle.setAttribute("cy", cy);
    circle.setAttribute("r", r);
    circle.setAttribute("fill", fillCol);
    circle.setAttribute("stroke", strokeCol);
    circle.setAttribute("stroke-width", "1.8");
    circle.setAttribute("stroke-dasharray", "5,4");
    svg.appendChild(circle);
  } else if (selectShapeMode === "polygon" && polygonPoints.length > 0) {
    const pts = [...polygonPoints, mouseCurrent];
    const pointsAttr = pts.map((pt) => `${pt.x},${pt.y}`).join(" ");
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("points", pointsAttr);
    poly.setAttribute("fill", fillCol);
    poly.setAttribute("stroke", strokeCol);
    poly.setAttribute("stroke-width", "1.8");
    poly.setAttribute("stroke-dasharray", "5,4");
    svg.appendChild(poly);

    for (const pt of polygonPoints) {
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", pt.x);
      dot.setAttribute("cy", pt.y);
      dot.setAttribute("r", "4");
      dot.setAttribute("fill", "#facc15");
      svg.appendChild(dot);
    }
  }
}

/**
 * 選択反転 (現在表示・未ロックの点群の選択状態を反転)
 */
function invertSelection() {
  if (!currentPoints || !pointCloud) return;
  const total = currentPoints.length / 3;
  const newSel = new Set();

  for (let i = 0; i < total; i++) {
    if (!isPointSelectable(i)) continue;
    if (!selectedIndices.has(i)) {
      newSel.add(i);
    }
  }

  selectedIndices = newSel;
  updatePointCloudDisplay();
  setStatusMessage(`選択を反転しました (${selectedIndices.size.toLocaleString()} 点を選択中)`);
}

function clearSelection() {
  if (!pointCloud) return;
  selectedIndices.clear();
  polygonPoints = [];
  brushStrokePoints = [];
  isSelecting = false;
  const boxElem = document.getElementById("selection-box");
  if (boxElem) boxElem.style.display = "none";
  const confirmBtn = document.getElementById("btn-confirm-polygon");
  if (confirmBtn) confirmBtn.disabled = true;
  if (isMeasureMode) {
    clearMeasure();
  }
  renderSelectionOverlay();
  updatePointCloudDisplay();
  setStatusMessage("選択を解除しました (Esc)");
}

/**
 * 3軸断面スライサーのUI同期と適用 (シーク基点位置 & 厚み指定モード対応)
 */
function syncSeekSlidersToActiveAxis() {
  const minSlider = document.getElementById("slider-seek-min-z");
  const maxSlider = document.getElementById("slider-seek-max-z");
  const modeSel = document.getElementById("select-seek-origin-mode");
  if (!minSlider || !maxSlider) return;

  const b = boundsXYZ[activeSeekAxis] || { min: 0, max: 1 };
  const s = seekRange[activeSeekAxis] || { min: b.min, max: b.max };
  const step = (b.max - b.min) / 400 || 0.02;

  minSlider.min = b.min;
  minSlider.max = b.max;
  minSlider.step = step;

  maxSlider.min = b.min;
  maxSlider.max = b.max;
  maxSlider.step = step;

  const mode = modeSel ? modeSel.value : "range";
  if (mode === "range") {
    minSlider.value = s.min;
    maxSlider.value = s.max;
    maxSlider.style.display = "";
  } else {
    const origin = s.origin !== undefined ? s.origin : 0.5 * (s.min + s.max);
    minSlider.value = origin;
    maxSlider.value = s.max;
    maxSlider.style.display = "none";
  }

  updateSeekRangeLabel();
}

function updateSeekRangeLabel() {
  const lbl = document.getElementById("label-seek-range");
  if (!lbl) return;
  const s = seekRange[activeSeekAxis];
  const b = boundsXYZ[activeSeekAxis];
  if (!s || !b) {
    lbl.textContent = "全域";
    return;
  }
  const axisUpper = activeSeekAxis.toUpperCase();
  if (Math.abs(s.min - b.min) < 1e-3 && Math.abs(s.max - b.max) < 1e-3) {
    lbl.textContent = `${axisUpper}: 全域`;
  } else {
    const thick = Math.max(0, s.max - s.min);
    const orig = s.origin !== undefined ? s.origin : 0.5 * (s.min + s.max);
    lbl.textContent = `${axisUpper}基点:${orig.toFixed(2)}m (厚:${thick.toFixed(2)}m)`;
  }
}

function applyHeightSeek() {
  if (!pointCloud || !currentPoints) return;

  const modeSel = document.getElementById("select-seek-origin-mode");
  const thickInput = document.getElementById("input-seek-thickness");
  const mode = modeSel ? modeSel.value : "range";
  const thick = Math.max(0.02, parseFloat(thickInput?.value) || 2.0);
  const b = boundsXYZ[activeSeekAxis] || { min: -100, max: 100 };

  let minVal = parseFloat(document.getElementById("slider-seek-min-z").value);
  let maxVal = parseFloat(document.getElementById("slider-seek-max-z").value);
  let originVal = minVal;

  if (mode === "center") {
    minVal = Math.max(b.min, originVal - thick * 0.5);
    maxVal = Math.min(b.max, originVal + thick * 0.5);
  } else if (mode === "plus") {
    minVal = Math.max(b.min, originVal);
    maxVal = Math.min(b.max, originVal + thick);
  } else if (mode === "minus") {
    minVal = Math.max(b.min, originVal - thick);
    maxVal = Math.min(b.max, originVal);
  } else {
    if (minVal > maxVal) {
      const tmp = minVal;
      minVal = maxVal;
      maxVal = tmp;
    }
    originVal = 0.5 * (minVal + maxVal);
  }

  seekRange[activeSeekAxis].min = minVal;
  seekRange[activeSeekAxis].max = maxVal;
  seekRange[activeSeekAxis].origin = originVal;
  updateSeekRangeLabel();
  updatePointCloudDisplay();
}

function setSeekOriginFromCurrentView() {
  if (!currentPoints) return;
  let fx = controls ? controls.target.x : 0;
  let fy = controls ? controls.target.y : 0;
  let fz = controls ? controls.target.z : 0;

  if (selectedIndices.size > 0) {
    let sx = 0, sy = 0, sz = 0, cnt = 0;
    for (const idx of selectedIndices) {
      sx += currentPoints[idx * 3];
      sy += currentPoints[idx * 3 + 1];
      sz += currentPoints[idx * 3 + 2];
      cnt++;
    }
    if (cnt > 0) {
      fx = sx / cnt;
      fy = sy / cnt;
      fz = sz / cnt;
    }
  }

  const coords = { x: fx, y: fy, z: fz };
  const targetVal = coords[activeSeekAxis];
  const minSlider = document.getElementById("slider-seek-min-z");
  if (minSlider) {
    minSlider.value = String(targetVal);
  }
  const modeSel = document.getElementById("select-seek-origin-mode");
  if (modeSel && modeSel.value === "range") {
    modeSel.value = "center";
  }
  syncSeekSlidersToActiveAxis();
  if (minSlider) minSlider.value = String(targetVal);
  applyHeightSeek();
  showToast(`🎯 ${activeSeekAxis.toUpperCase()}軸のシーク基点を現在位置 (${targetVal.toFixed(2)}m) にセットしました`);
}

function resetAllSeekRanges() {
  if (!currentPoints) return;
  seekRange = {
    x: { min: boundsXYZ.x.min, max: boundsXYZ.x.max },
    y: { min: boundsXYZ.y.min, max: boundsXYZ.y.max },
    z: { min: boundsXYZ.z.min, max: boundsXYZ.z.max },
  };
  syncSeekSlidersToActiveAxis();
  updatePointCloudDisplay();
  setStatusMessage("XYZすべての断面スライサーを全域表示にリセットしました");
}

/**
 * 断面ボックスの外側にある未ロック有効点を一括でノイズレイヤーへ移動（断面外クロップ）
 */
function cropOutsideSection() {
  if (!currentPoints || !layerMask) return;
  const n = currentPoints.length / 3;
  const eps = 1e-4;
  let cropCount = 0;

  for (let i = 0; i < n; i++) {
    if (!isPointUnlockedNormal(i)) continue;
    const x = currentPoints[i * 3];
    const y = currentPoints[i * 3 + 1];
    const z = currentPoints[i * 3 + 2];
    const inside =
      x >= seekRange.x.min - eps && x <= seekRange.x.max + eps &&
      y >= seekRange.y.min - eps && y <= seekRange.y.max + eps &&
      z >= seekRange.z.min - eps && z <= seekRange.z.max + eps;
    if (!inside) cropCount++;
  }

  if (cropCount === 0) {
    setStatusMessage("現在の断面範囲外に移動可能な未ロック有効点はありません");
    return;
  }

  pushUndoState();

  for (let i = 0; i < n; i++) {
    if (!isPointUnlockedNormal(i)) continue;
    const x = currentPoints[i * 3];
    const y = currentPoints[i * 3 + 1];
    const z = currentPoints[i * 3 + 2];
    const inside =
      x >= seekRange.x.min - eps && x <= seekRange.x.max + eps &&
      y >= seekRange.y.min - eps && y <= seekRange.y.max + eps &&
      z >= seekRange.z.min - eps && z <= seekRange.z.max + eps;
    if (!inside) {
      layerMask[i] = 0;
    }
  }

  selectedIndices.clear();
  resetAllSeekRanges();
  setStatusMessage(`断面ボックス外の ${cropCount.toLocaleString()} 点をノイズレイヤーへ一括移動しました`);
}

/**
 * 統計的外れ値除去 (SOR) - 未ロックの有効レイヤー点を対象に外れ値をノイズレイヤーへ移動
 */
function runSOR() {
  if (!currentPoints || !layerMask) return;
  showLoading("SOR自動ノイズ除去を実行中...");

  setTimeout(() => {
    const stdRatio = parseFloat(document.getElementById("slider-sor-std").value) || 2.0;
    const total = currentPoints.length / 3;

    const normalIndices = [];
    for (let i = 0; i < total; i++) {
      if (isPointUnlockedNormal(i)) normalIndices.push(i);
    }

    if (normalIndices.length === 0) {
      hideLoading();
      return;
    }

    const meanDists = new Float32Array(normalIndices.length);
    let sumMean = 0;
    const step = Math.max(1, Math.floor(normalIndices.length / 300));

    for (let idx = 0; idx < normalIndices.length; idx++) {
      const i = normalIndices[idx];
      const xi = currentPoints[i * 3];
      const yi = currentPoints[i * 3 + 1];
      const zi = currentPoints[i * 3 + 2];

      let dSum = 0;
      let count = 0;
      for (let jIdx = 0; jIdx < normalIndices.length; jIdx += step) {
        const j = normalIndices[jIdx];
        if (i === j) continue;
        const dx = xi - currentPoints[j * 3];
        const dy = yi - currentPoints[j * 3 + 1];
        const dz = zi - currentPoints[j * 3 + 2];
        dSum += Math.sqrt(dx * dx + dy * dy + dz * dz);
        count++;
      }
      const avg = count > 0 ? dSum / count : 0;
      meanDists[idx] = avg;
      sumMean += avg;
    }

    const mean = sumMean / normalIndices.length;
    let varSum = 0;
    for (let idx = 0; idx < normalIndices.length; idx++) {
      varSum += (meanDists[idx] - mean) ** 2;
    }
    const std = Math.sqrt(varSum / normalIndices.length);
    const threshold = mean + stdRatio * std;

    const outlierIndices = [];
    for (let idx = 0; idx < normalIndices.length; idx++) {
      if (meanDists[idx] > threshold) {
        outlierIndices.push(normalIndices[idx]);
      }
    }

    hideLoading();
    if (outlierIndices.length > 0) {
      pushUndoState();

      // 自動除去用個別レイヤー「自動除去: SOR」へ振り分け
      const autoLayerName = "自動除去: SOR";
      let autoLid = null;
      for (const [k, v] of Object.entries(layersConfig)) {
        if (v.name === autoLayerName) {
          autoLid = Number(k);
          break;
        }
      }
      if (autoLid === null) {
        autoLid = Object.keys(layersConfig).length > 0
          ? Math.max(...Object.keys(layersConfig).map(Number)) + 1
          : 10;
        layersConfig[autoLid] = {
          id: autoLid,
          name: autoLayerName,
          visible: false, // 自動除去ノイズレイヤーは初期状態で非表示(画面・出力から除外)
          locked: false,
          color: "#f97316", // 視認性の高いオレンジ色
          useLayerColor: true,
          isNoise: true,
        };
      }

      for (const i of outlierIndices) {
        layerMask[i] = autoLid;
      }
      selectedIndices.clear();
      rebuildBaseColors();
      updatePointCloudDisplay();
      updateUndoRedoUI();
      setStatusMessage(`SORにより ${outlierIndices.length.toLocaleString()} 点の外れ値を「${autoLayerName}」へ分離しました`);
      showToast(`🪄 自動除去: ${outlierIndices.length.toLocaleString()} 点を「${autoLayerName}」へ分離`);
    } else {
      setStatusMessage("外れ値ノイズは検出されませんでした");
    }
  }, 50);
}

/**
 * 指定された条件関数 predicate(i) を満たす点群を1つのPLYファイルとしてダウンロード出力
 */
function downloadPLYByPredicate(predicate, filename) {
  const total = currentPoints.length / 3;
  let validCount = 0;
  for (let i = 0; i < total; i++) {
    if (predicate(i)) validCount++;
  }
  if (validCount === 0) return 0;

  let header = "ply\n";
  header += "format ascii 1.0\n";
  header += `element vertex ${validCount}\n`;
  header += "property double x\n";
  header += "property double y\n";
  header += "property double z\n";
  header += "property uchar red\n";
  header += "property uchar green\n";
  header += "property uchar blue\n";
  header += "end_header\n";

  const chunks = [header];
  let lineBuf = "";
  const exportColorSource = rawRgbColors || baseColors;

  for (let i = 0; i < total; i++) {
    if (!predicate(i)) continue;
    const x = currentPoints[i * 3];
    const y = currentPoints[i * 3 + 1];
    const z = currentPoints[i * 3 + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    const rc = exportColorSource ? exportColorSource[i * 3] : 0.2;
    const gc = exportColorSource ? exportColorSource[i * 3 + 1] : 0.75;
    const bc = exportColorSource ? exportColorSource[i * 3 + 2] : 0.95;
    const r = Math.min(255, Math.max(0, Math.round((Number.isFinite(rc) ? rc : 0) * 255)));
    const g = Math.min(255, Math.max(0, Math.round((Number.isFinite(gc) ? gc : 0) * 255)));
    const b = Math.min(255, Math.max(0, Math.round((Number.isFinite(bc) ? bc : 0) * 255)));
    lineBuf += `${x} ${y} ${z} ${r} ${g} ${b}\n`;
    if (lineBuf.length > 65536) {
      chunks.push(lineBuf);
      lineBuf = "";
    }
  }
  if (lineBuf.length > 0) chunks.push(lineBuf);

  const blob = new Blob(chunks, { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
  return validCount;
}

/**
 * ④ PLY形式エクスポート (有効レイヤー統合 / 表示中のみ / 単一レイヤー / ノイズのみ / 全レイヤー分割対応)
 */
function exportPLY() {
  if (!currentPoints || currentPoints.length === 0 || !layerMask) {
    alert("エクスポートする点群データがありません。");
    return;
  }

  const mode = document.getElementById("select-export-mode")?.value || "normal";

  if (mode === "split_all") {
    let exportedFiles = 0;
    let totalPts = 0;
    for (const [lidStr, lcfg] of Object.entries(layersConfig)) {
      const lid = Number(lidStr);
      if (isNoiseLayer(lid)) continue;
      const safeName = lcfg.name.replace(/[\\/:*?"<>|\s]+/g, "_");
      const cnt = downloadPLYByPredicate((i) => layerMask[i] === lid, `pointcloud_${safeName}.ply`);
      if (cnt > 0) {
        exportedFiles++;
        totalPts += cnt;
      }
    }
    if (exportedFiles === 0) {
      alert("分割出力可能な有効レイヤーの点がありません。");
      return;
    }
    setStatusMessage(`${exportedFiles} 個のレイヤー別PLYファイル (計 ${totalPts.toLocaleString()} 点) を分割出力しました`);
    return;
  }

  let predicate;
  let filename = "pointcloud_cleaned.ply";
  if (mode === "visible") {
    predicate = (i) => isPointVisible(i);
    filename = "pointcloud_visible.ply";
  } else if (mode === "single_layer") {
    const targetId = Number(document.getElementById("select-target-layer")?.value ?? 1);
    const lcfg = layersConfig[targetId];
    const safeName = (lcfg ? lcfg.name : `layer_${targetId}`).replace(/[\\/:*?"<>|\s]+/g, "_");
    predicate = (i) => layerMask[i] === targetId;
    filename = `pointcloud_${safeName}.ply`;
  } else if (mode === "noise_only") {
    predicate = (i) => isNoiseLayer(layerMask[i]);
    filename = "pointcloud_noise.ply";
  } else {
    predicate = (i) => !isNoiseLayer(layerMask[i]);
    filename = "pointcloud_cleaned.ply";
  }

  const validCount = downloadPLYByPredicate(predicate, filename);
  if (validCount === 0) {
    alert("選択された出力モードに該当する点が 0 点のため出力できません。");
    return;
  }
  setStatusMessage(`${validCount.toLocaleString()} 点を ${filename} として出力しました`);
}

/**
 * テキスト点群パース (倍精度 Float64Array で元の座標値を完全保持)
 */
function parseTextPointCloud(text) {
  const lines = text.trim().split(/\r?\n/);
  const pts = [];
  const cols = [];
  let inPlyHeader = false;

  for (let line of lines) {
    line = line.trim();
    if (!line || line.startsWith("#") || line.startsWith("//")) continue;
    if (line === "ply") {
      inPlyHeader = true;
      continue;
    }
    if (inPlyHeader) {
      if (line === "end_header") {
        inPlyHeader = false;
      }
      continue;
    }
    if (
      line.startsWith("format") ||
      line.startsWith("comment") ||
      line.startsWith("element") ||
      line.startsWith("property") ||
      line.startsWith("obj_info") ||
      line.startsWith("end_header")
    ) {
      continue;
    }

    const parts = line.split(/[\s,]+/);
    if (parts.length >= 3) {
      const x = Number(parts[0]);
      const y = Number(parts[1]);
      const z = Number(parts[2]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      pts.push(x, y, z);

      // 7列以上 (PTS標準: X Y Z Intensity R G B) または 6列 (X Y Z R G B) に対応
      const rIdx = parts.length >= 7 ? 4 : 3;
      if (parts.length >= rIdx + 3) {
        const rv = Number(parts[rIdx]);
        const gv = Number(parts[rIdx + 1]);
        const bv = Number(parts[rIdx + 2]);
        if (Number.isFinite(rv) && Number.isFinite(gv) && Number.isFinite(bv)) {
          const maxC = Math.max(rv, gv, bv);
          const scale = maxC > 255 ? 65535 : (maxC > 1 ? 255 : 1);
          cols.push(
            Math.min(1, Math.max(0, rv / scale)),
            Math.min(1, Math.max(0, gv / scale)),
            Math.min(1, Math.max(0, bv / scale))
          );
        }
      }
    }
  }

  return {
    points: new Float64Array(pts),
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

  // 🧭 2点ピック斜め壁アライン
  document.getElementById("btn-wall-pick")?.addEventListener("click", () => toggleWallPickMode());
  document.getElementById("btn-wall-horiz")?.addEventListener("click", () => applyWallAlignedView("horiz"));
  document.getElementById("btn-wall-vert")?.addEventListener("click", () => applyWallAlignedView("vert"));
  document.getElementById("btn-wall-ortho")?.addEventListener("click", () => applyWallAlignedView("ortho_front"));
  document.getElementById("btn-wall-edge")?.addEventListener("click", () => applyWallAlignedView("edge"));
  document.getElementById("btn-wall-flip")?.addEventListener("click", () => flipWallAlignedView());

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
    syncModeSegments();
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
    if (mode === "rgb" && (!rawRgbColors || rawRgbColors.length !== currentPoints.length)) {
      setStatusMessage("この点群にはRGB色情報がないため、高さカラーで表示しています");
    }
    rebuildBaseColors();
    updatePointCloudDisplay();
  });

  // レイヤー表示切替 & マルチレイヤー操作 & 自動分類 & 状態保存
  const chkLayerNormal = document.getElementById("chk-layer-normal");
  const chkLayerNoise = document.getElementById("chk-layer-noise");
  if (chkLayerNormal) {
    chkLayerNormal.addEventListener("change", (e) => {
      showNormalLayer = e.target.checked;
      for (const [lidStr, lcfg] of Object.entries(layersConfig)) {
        if (!isNoiseLayer(Number(lidStr))) lcfg.visible = showNormalLayer;
      }
      updatePointCloudDisplay();
    });
  }
  if (chkLayerNoise) {
    chkLayerNoise.addEventListener("change", (e) => {
      showNoiseLayer = e.target.checked;
      for (const [lidStr, lcfg] of Object.entries(layersConfig)) {
        if (isNoiseLayer(Number(lidStr))) lcfg.visible = showNoiseLayer;
      }
      updatePointCloudDisplay();
    });
  }
  const btnAddCustomLayer = document.getElementById("btn-add-custom-layer");
  if (btnAddCustomLayer) btnAddCustomLayer.addEventListener("click", addCustomLayer);

  const btnMoveToLayer = document.getElementById("btn-move-to-layer");
  if (btnMoveToLayer) btnMoveToLayer.addEventListener("click", moveSelectedToTargetLayer);

  const btnAutoGround = document.getElementById("btn-auto-ground");
  if (btnAutoGround) btnAutoGround.addEventListener("click", autoExtractGround);

  const btnAutoPlane = document.getElementById("btn-auto-plane");
  if (btnAutoPlane) btnAutoPlane.addEventListener("click", autoExtractPlane);

  const btnSaveProj = document.getElementById("btn-save-layer-proj");
  if (btnSaveProj) btnSaveProj.addEventListener("click", saveLayerProjectJSON);

  const btnLoadProj = document.getElementById("btn-load-layer-proj");
  const inputLayerProjFile = document.getElementById("input-layer-proj-file");
  if (btnLoadProj && inputLayerProjFile) {
    btnLoadProj.addEventListener("click", () => inputLayerProjFile.click());
    inputLayerProjFile.addEventListener("change", (e) => {
      if (e.target.files.length > 0) loadLayerProjectJSON(e.target.files[0]);
      e.target.value = "";
    });
  }

  const btnRestoreSelected = document.getElementById("btn-restore-selected-noise");
  if (btnRestoreSelected) {
    btnRestoreSelected.addEventListener("click", restoreSelectedNoise);
  }
  const btnRestoreAll = document.getElementById("btn-restore-all-noise");
  if (btnRestoreAll) {
    btnRestoreAll.addEventListener("click", restoreAllNoise);
  }

  // 選択形状セレクト (brush / rect / circle / polygon)
  const selectShapeElem = document.getElementById("select-shape-mode");
  if (selectShapeElem) {
    selectShapeElem.addEventListener("change", (e) => {
      selectShapeMode = e.target.value;
      polygonPoints = [];
      brushStrokePoints = [];
      const confirmBtn = document.getElementById("btn-confirm-polygon");
      if (confirmBtn) confirmBtn.disabled = true;
      renderSelectionOverlay();
      if (isSelectMode) {
        if (selectShapeMode === "polygon") {
          setStatusMessage("【多角形選択】画面上を左クリックして頂点を追加し、Enterまたは右クリックで確定");
        } else if (selectShapeMode === "circle") {
          setStatusMessage("【円形選択】中心から左ドラッグして円形範囲を選択");
        } else if (selectShapeMode === "brush") {
          setStatusMessage("【ブラシ選択/消去】画面上を左ドラッグでなぞって選択・消去 ([ / ]キーでサイズ変更)");
        } else {
          setStatusMessage("【矩形選択】左ドラッグして四角形範囲を選択");
        }
      }
      syncModeSegments();
    });
  }

  // ブラシ半径スライダー
  const brushSlider = document.getElementById("slider-brush-radius");
  if (brushSlider) {
    brushSlider.addEventListener("input", (e) => {
      brushRadius = Math.max(6, Math.min(150, parseInt(e.target.value, 10) || 28));
      const lbl = document.getElementById("label-brush-radius");
      if (lbl) lbl.textContent = `${brushRadius} px`;
      renderSelectionOverlay();
    });
  }

  // 即時削除モード & ノイズ専用モード切替
  document.getElementById("btn-instant-del")?.addEventListener("click", () => toggleInstantDeleteMode());
  document.getElementById("chk-instant-del")?.addEventListener("click", () => toggleInstantDeleteMode());
  document.getElementById("btn-clean-mode")?.addEventListener("click", () => toggleCleanFocusMode());
  document.getElementById("btn-clean-mode-side")?.addEventListener("click", () => toggleCleanFocusMode());

  // スマートノイズ処理コマンド4種
  document.getElementById("btn-magic-wand")?.addEventListener("click", magicWandClusterSelect);
  document.getElementById("btn-local-clean")?.addEventListener("click", localFineCleanSelected);
  document.getElementById("btn-keep-main-cluster")?.addEventListener("click", keepMainClusterOnly);
  document.getElementById("btn-quick-edge-clean")?.addEventListener("click", quickEdgeVeilClean);

  // ワンクリック・プリセット3種
  document.getElementById("btn-preset-rough")?.addEventListener("click", () => applyFilterPreset("rough"));
  document.getElementById("btn-preset-std")?.addEventListener("click", () => applyFilterPreset("standard"));
  document.getElementById("btn-preset-fine")?.addEventListener("click", () => applyFilterPreset("fine"));

  // 手動選択モード切替
  const selectToggleBtn = document.getElementById("btn-toggle-select");
  selectToggleBtn.addEventListener("click", () => {
    isSelectMode = !isSelectMode;
    controls.enabled = !isSelectMode;
    polygonPoints = [];
    const confirmBtn = document.getElementById("btn-confirm-polygon");
    if (confirmBtn) confirmBtn.disabled = true;
    renderSelectionOverlay();

    if (isSelectMode) {
      if (isMeasureMode) measureToggleBtn.click(); // 計測モードと排他
      selectToggleBtn.textContent = "🔲 手動選択モード: ON";
      selectToggleBtn.classList.add("btn-active");
      if (selectShapeMode === "polygon") {
        setStatusMessage("【多角形選択】頂点を左クリックして追加、Enterキーで確定 (Shift追加 / Ctrl除外)");
      } else if (selectShapeMode === "circle") {
        setStatusMessage("【円形選択】中心から左ドラッグして選択 (Shift追加 / Ctrl除外)");
      } else {
        setStatusMessage("【矩形選択】画面上を左ドラッグして点を囲んでください (Shift追加 / Ctrl除外)");
      }
    } else {
      selectToggleBtn.textContent = "🔲 手動選択モード: OFF";
      selectToggleBtn.classList.remove("btn-active");
      clearSelection();
    }
    syncModeSegments();
  });

  const btnConfirmPolygon = document.getElementById("btn-confirm-polygon");
  if (btnConfirmPolygon) {
    btnConfirmPolygon.addEventListener("click", (e) => {
      confirmPolygonSelection(e.shiftKey, e.ctrlKey || e.metaKey);
    });
  }

  const btnInvertSel = document.getElementById("btn-invert-selection");
  if (btnInvertSel) {
    btnInvertSel.addEventListener("click", invertSelection);
  }

  // マウスクリック判定 (距離計測 / 矩形・円形・多角形・ブラシ選択)
  const boxElem = document.getElementById("selection-box");

  window.addEventListener("mousedown", (e) => {
    if (e.target.closest(".sidebar") || e.target.closest(".top-nav") || e.target.closest(".bottom-bar") || e.target.closest(".modal-card") || e.target.closest(".bottom-overlay-bar") || e.target.closest("#context-menu")) return;

    // 🧭 2点ピック斜め壁アラインモード時のクリック
    if (isWallPickMode && e.button === 0) {
      handleWallPickClick(e.clientX, e.clientY);
      return;
    }

    // 距離計測モード時のクリック
    if (isMeasureMode && e.button === 0) {
      handleMeasureClick(e.clientX, e.clientY);
      return;
    }

    // Spaceキー長押し中 または 手動選択モード時でも Alt+左ドラッグ・右/中ドラッグ はカメラ操作を許可
    if (spacePanActive) {
      controls.enabled = true;
      controls.mouseButtons.LEFT = THREE.MOUSE.PAN;
      return;
    }
    if (isSelectMode && (e.altKey || e.button === 1 || (e.button === 2 && selectShapeMode !== "polygon"))) {
      controls.enabled = true;
      return;
    }

    // 手動選択モード時
    if (isSelectMode) {
      controls.enabled = false;
      if (selectShapeMode === "polygon") {
        if (e.button === 0) {
          polygonPoints.push({ x: e.clientX, y: e.clientY });
          mouseCurrent.x = e.clientX;
          mouseCurrent.y = e.clientY;
          if (btnConfirmPolygon) btnConfirmPolygon.disabled = polygonPoints.length < 3;
          renderSelectionOverlay();
          setStatusMessage(`多角形頂点 ${polygonPoints.length} 個目を追加 (3点以上でEnterまたは確定ボタン)`);
        } else if (e.button === 2 && polygonPoints.length >= 3) {
          e.preventDefault();
          confirmPolygonSelection(e.shiftKey, e.ctrlKey || e.metaKey);
        }
        return;
      }

      if (e.button === 0) {
        isSelecting = true;
        selectStart.x = e.clientX;
        selectStart.y = e.clientY;
        selectEnd.x = e.clientX;
        selectEnd.y = e.clientY;

        if (selectShapeMode === "rect") {
          boxElem.style.left = `${selectStart.x}px`;
          boxElem.style.top = `${selectStart.y}px`;
          boxElem.style.width = "0px";
          boxElem.style.height = "0px";
          boxElem.style.display = "block";
        } else if (selectShapeMode === "circle") {
          renderSelectionOverlay();
        } else if (selectShapeMode === "brush") {
          brushStrokePoints = [{ x: e.clientX, y: e.clientY }];
          mouseCurrent.x = e.clientX;
          mouseCurrent.y = e.clientY;
          renderSelectionOverlay();
        }
      }
    }
  });

  window.addEventListener("mousemove", (e) => {
    mouseCurrent.x = e.clientX;
    mouseCurrent.y = e.clientY;

    if (isSelectMode && selectShapeMode === "polygon" && polygonPoints.length > 0) {
      renderSelectionOverlay();
      return;
    }

    if (isSelectMode && selectShapeMode === "brush" && !isSelecting) {
      renderSelectionOverlay();
      return;
    }

    if (!isSelecting) return;
    selectEnd.x = e.clientX;
    selectEnd.y = e.clientY;

    if (selectShapeMode === "rect") {
      const left = Math.min(selectStart.x, selectEnd.x);
      const top = Math.min(selectStart.y, selectEnd.y);
      const width = Math.abs(selectStart.x - selectEnd.x);
      const height = Math.abs(selectStart.y - selectEnd.y);

      boxElem.style.left = `${left}px`;
      boxElem.style.top = `${top}px`;
      boxElem.style.width = `${width}px`;
      boxElem.style.height = `${height}px`;
    } else if (selectShapeMode === "circle") {
      renderSelectionOverlay();
    } else if (selectShapeMode === "brush") {
      const lastPt = brushStrokePoints[brushStrokePoints.length - 1];
      if (!lastPt || Math.hypot(e.clientX - lastPt.x, e.clientY - lastPt.y) >= 3) {
        brushStrokePoints.push({ x: e.clientX, y: e.clientY });
      }
      renderSelectionOverlay();
    }
  });

  window.addEventListener("mouseup", (e) => {
    if (isSelectMode && !isSelecting) {
      controls.enabled = false;
    }
    if (!isSelecting) return;
    isSelecting = false;
    boxElem.style.display = "none";

    const shiftKey = e.shiftKey;
    const ctrlKey = e.ctrlKey || e.metaKey;

    if (selectShapeMode === "rect") {
      renderSelectionOverlay();
      const left = Math.min(selectStart.x, selectEnd.x);
      const top = Math.min(selectStart.y, selectEnd.y);
      const right = Math.max(selectStart.x, selectEnd.x);
      const bottom = Math.max(selectStart.y, selectEnd.y);

      if (right - left > 5 && bottom - top > 5) {
        processSelectionBox({ left, top, right, bottom }, shiftKey, ctrlKey);
      }
    } else if (selectShapeMode === "circle") {
      renderSelectionOverlay();
      const radius = Math.hypot(selectEnd.x - selectStart.x, selectEnd.y - selectStart.y);
      if (radius > 5) {
        processSelectionCircle(selectStart.x, selectStart.y, radius, shiftKey, ctrlKey);
      }
    } else if (selectShapeMode === "brush") {
      const stroke = brushStrokePoints.slice();
      brushStrokePoints = [];
      renderSelectionOverlay();
      if (stroke.length > 0) {
        processSelectionBrush(stroke, brushRadius, shiftKey, ctrlKey);
      }
    }
  });

  // Ctrl + ホイールでブラシ半径を直感調整
  window.addEventListener("wheel", (e) => {
    if (isSelectMode && selectShapeMode === "brush" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      const step = e.deltaY < 0 ? 4 : -4;
      brushRadius = Math.max(6, Math.min(150, brushRadius + step));
      const slider = document.getElementById("slider-brush-radius");
      const lbl = document.getElementById("label-brush-radius");
      if (slider) slider.value = String(brushRadius);
      if (lbl) lbl.textContent = `${brushRadius} px`;
      renderSelectionOverlay();
    }
  }, { passive: false });

  // ダブルクリックで中心移動（多角形選択中は確定）
  window.addEventListener("dblclick", (e) => {
    if (e.target.closest(".sidebar") || e.target.closest(".top-nav") || e.target.closest(".bottom-bar") || e.target.closest(".modal-card")) return;
    if (isSelectMode && selectShapeMode === "polygon" && polygonPoints.length >= 3) {
      confirmPolygonSelection(e.shiftKey, e.ctrlKey || e.metaKey);
      return;
    }
    focusOnPoint(e.clientX, e.clientY);
  });

  document.getElementById("btn-delete-selected").addEventListener("click", deleteSelectedPoints);
  document.getElementById("btn-clear-selection").addEventListener("click", clearSelection);

  document.getElementById("btn-undo").addEventListener("click", undo);
  document.getElementById("btn-redo").addEventListener("click", redo);

  // ショートカットキー (V: 回転/パントグル, H: パン, B: ブラシ, R: 矩形, C: 円形, L: 多角形, N: 専用モード, G: 連結塊選択, [/]: ブラシサイズ, Space: 一時パン)
  window.addEventListener("keydown", (e) => {
    keysPressed[e.code] = true;

    if (isInputFocused()) return;

    if (e.code === "Space" && !spacePanActive) {
      e.preventDefault();
      spacePanActive = true;
      controls.enabled = true;
      controls.mouseButtons.LEFT = THREE.MOUSE.PAN;
      document.body.style.cursor = "grab";
      return;
    }

    if (e.key === "v" || e.key === "V") {
      if (!isSelectMode && !isMeasureMode && !isPanMode) {
        setInteractionMode("pan");
      } else {
        setInteractionMode("orbit");
      }
    } else if (e.key === "h" || e.key === "H") {
      setInteractionMode("pan");
    } else if (e.key === "b" || e.key === "B") {
      setInteractionMode("brush");
    } else if (e.key === "r" || e.key === "R") {
      setInteractionMode("rect");
    } else if (e.key === "c" || e.key === "C") {
      setInteractionMode("circle");
    } else if (e.key === "l" || e.key === "L") {
      setInteractionMode("poly");
    } else if (e.key === "n" || e.key === "N") {
      toggleCleanFocusMode();
    } else if (e.key === "g" || e.key === "G") {
      magicWandClusterSelect();
    } else if (e.key === "[" || e.key === "]") {
      const step = e.key === "]" ? 4 : -4;
      brushRadius = Math.max(6, Math.min(150, brushRadius + step));
      const slider = document.getElementById("slider-brush-radius");
      const lbl = document.getElementById("label-brush-radius");
      if (slider) slider.value = String(brushRadius);
      if (lbl) lbl.textContent = `${brushRadius} px`;
      renderSelectionOverlay();
      setStatusMessage(`🧹 ブラシ半径: ${brushRadius} px`);
    } else if (e.key === "p" || e.key === "P" || e.code === "Numpad5") {
      toggleProjectionMode();
    } else if (e.key === "f" || e.key === "F") {
      fitView();
    } else if (e.key === "u" || e.key === "U") {
      toggleWallPickMode();
    } else if (e.key === "m" || e.key === "M") {
      document.getElementById("btn-toggle-measure").click();
    } else if (e.key === "i" || e.key === "I") {
      invertSelection();
    } else if (e.key === "Enter" && isSelectMode && selectShapeMode === "polygon" && polygonPoints.length >= 3) {
      confirmPolygonSelection(e.shiftKey, e.ctrlKey || e.metaKey);
    } else if (e.key === "Escape") {
      if (isWallPickMode) toggleWallPickMode(false);
      clearSelection();
      hideContextMenu();
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
    if (e.code === "Space" && spacePanActive) {
      spacePanActive = false;
      controls.mouseButtons.LEFT = isPanMode ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
      controls.enabled = !isSelectMode;
      document.body.style.cursor = isPanMode ? "grab" : "";
    }
  });

  // 右クリック文脈メニュー
  window.addEventListener("contextmenu", (e) => {
    if (e.target.closest(".sidebar") || e.target.closest(".top-nav") || e.target.closest(".bottom-bar") || e.target.closest(".modal-card") || e.target.closest(".bottom-overlay-bar")) {
      return;
    }
    // 多角形選択モードで3点以上の場合は多角形確定を優先
    if (isSelectMode && selectShapeMode === "polygon" && polygonPoints.length >= 3) {
      e.preventDefault();
      confirmPolygonSelection(e.shiftKey, e.ctrlKey || e.metaKey);
      return;
    }
    e.preventDefault();
    showContextMenu(e.clientX, e.clientY);
  });

  window.addEventListener("click", (e) => {
    if (!e.target.closest("#context-menu")) {
      hideContextMenu();
    }
  });

  // サイドバー格納トグル
  document.getElementById("btn-toggle-sidebar")?.addEventListener("click", toggleSidebar);

  // ワークフロー切替タブ
  const flowTabs = document.querySelectorAll("#sidebar-nav-tabs .flow-tab-btn");
  flowTabs.forEach((tabBtn) => {
    tabBtn.addEventListener("click", () => {
      setWorkflowTab(tabBtn.dataset.tab);
    });
  });

  // 1クリック操作モード切替セグメント (上部バー & 左パネル)
  document.getElementById("seg-btn-orbit")?.addEventListener("click", () => setInteractionMode("orbit"));
  document.getElementById("seg-btn-pan")?.addEventListener("click", () => setInteractionMode("pan"));
  document.getElementById("seg-btn-brush")?.addEventListener("click", () => setInteractionMode("brush"));
  document.getElementById("seg-btn-rect")?.addEventListener("click", () => setInteractionMode("rect"));
  document.getElementById("seg-btn-circle")?.addEventListener("click", () => setInteractionMode("circle"));
  document.getElementById("seg-btn-poly")?.addEventListener("click", () => setInteractionMode("poly"));
  document.getElementById("seg-btn-measure")?.addEventListener("click", () => setInteractionMode("measure"));

  document.getElementById("side-btn-orbit")?.addEventListener("click", () => setInteractionMode("orbit"));
  document.getElementById("side-btn-pan")?.addEventListener("click", () => setInteractionMode("pan"));
  document.getElementById("side-btn-brush")?.addEventListener("click", () => setInteractionMode("brush"));
  document.getElementById("side-btn-rect")?.addEventListener("click", () => setInteractionMode("rect"));
  document.getElementById("side-btn-circle")?.addEventListener("click", () => setInteractionMode("circle"));
  document.getElementById("side-btn-poly")?.addEventListener("click", () => setInteractionMode("poly"));

  // フローティング・アクションバー
  document.getElementById("floating-btn-delete")?.addEventListener("click", deleteSelectedPoints);
  document.getElementById("floating-btn-wand")?.addEventListener("click", magicWandClusterSelect);
  document.getElementById("floating-btn-local-clean")?.addEventListener("click", localFineCleanSelected);
  document.getElementById("floating-btn-move")?.addEventListener("click", moveSelectedToTargetLayer);
  document.getElementById("floating-btn-restore")?.addEventListener("click", restoreSelectedNoise);
  document.getElementById("floating-btn-invert")?.addEventListener("click", invertSelection);
  document.getElementById("floating-btn-clear")?.addEventListener("click", clearSelection);

  // 右クリック文脈メニュー各項目
  document.getElementById("ctx-item-delete")?.addEventListener("click", () => { hideContextMenu(); deleteSelectedPoints(); });
  document.getElementById("ctx-item-wand")?.addEventListener("click", () => { hideContextMenu(); magicWandClusterSelect(); });
  document.getElementById("ctx-item-keep-main")?.addEventListener("click", () => { hideContextMenu(); keepMainClusterOnly(); });
  document.getElementById("ctx-item-edge-clean")?.addEventListener("click", () => { hideContextMenu(); quickEdgeVeilClean(); });
  document.getElementById("ctx-item-restore")?.addEventListener("click", () => { hideContextMenu(); restoreSelectedNoise(); });
  document.getElementById("ctx-item-invert")?.addEventListener("click", () => { hideContextMenu(); invertSelection(); });
  document.getElementById("ctx-item-clear")?.addEventListener("click", () => { hideContextMenu(); clearSelection(); });

  document.getElementById("ctx-mode-orbit")?.addEventListener("click", () => { hideContextMenu(); setInteractionMode("orbit"); });
  document.getElementById("ctx-mode-pan")?.addEventListener("click", () => { hideContextMenu(); setInteractionMode("pan"); });
  document.getElementById("ctx-mode-brush")?.addEventListener("click", () => { hideContextMenu(); setInteractionMode("brush"); });
  document.getElementById("ctx-mode-rect")?.addEventListener("click", () => { hideContextMenu(); setInteractionMode("rect"); });
  document.getElementById("ctx-mode-circle")?.addEventListener("click", () => { hideContextMenu(); setInteractionMode("circle"); });
  document.getElementById("ctx-mode-poly")?.addEventListener("click", () => { hideContextMenu(); setInteractionMode("poly"); });
  document.getElementById("ctx-mode-measure")?.addEventListener("click", () => { hideContextMenu(); setInteractionMode("measure"); });

  document.getElementById("ctx-view-fit")?.addEventListener("click", () => { hideContextMenu(); fitView(); });
  document.getElementById("ctx-view-proj")?.addEventListener("click", () => { hideContextMenu(); toggleProjectionMode(); });
  document.getElementById("ctx-item-undo")?.addEventListener("click", () => { hideContextMenu(); undo(); });
  document.getElementById("ctx-item-redo")?.addEventListener("click", () => { hideContextMenu(); redo(); });

  // 自動除去 1つ戻る/進む
  document.getElementById("btn-auto-undo")?.addEventListener("click", undo);
  document.getElementById("btn-auto-redo")?.addEventListener("click", redo);

  document.getElementById("slider-sor-k").addEventListener("input", (e) => {
    document.getElementById("label-sor-k").textContent = e.target.value;
  });
  document.getElementById("slider-sor-std").addEventListener("input", (e) => {
    document.getElementById("label-sor-std").textContent = parseFloat(e.target.value).toFixed(1);
  });
  document.getElementById("btn-run-sor").addEventListener("click", runSOR);

  document.getElementById("btn-export-ply").addEventListener("click", exportPLY);

  // 3軸断面スライサー & 断面外クロップ (シーク基点位置・厚み指定対応)
  const seekAxisSelect = document.getElementById("select-seek-axis");
  if (seekAxisSelect) {
    seekAxisSelect.addEventListener("change", (e) => {
      activeSeekAxis = e.target.value;
      syncSeekSlidersToActiveAxis();
    });
  }
  const seekOriginModeSelect = document.getElementById("select-seek-origin-mode");
  if (seekOriginModeSelect) {
    seekOriginModeSelect.addEventListener("change", () => {
      syncSeekSlidersToActiveAxis();
      applyHeightSeek();
    });
  }
  const seekThicknessInput = document.getElementById("input-seek-thickness");
  if (seekThicknessInput) {
    seekThicknessInput.addEventListener("input", applyHeightSeek);
    seekThicknessInput.addEventListener("change", applyHeightSeek);
  }
  document.getElementById("btn-seek-from-current")?.addEventListener("click", setSeekOriginFromCurrentView);
  document.getElementById("slider-seek-min-z").addEventListener("input", applyHeightSeek);
  document.getElementById("slider-seek-max-z").addEventListener("input", applyHeightSeek);
  const btnCropSection = document.getElementById("btn-crop-section");
  if (btnCropSection) {
    btnCropSection.addEventListener("click", cropOutsideSection);
  }
  document.getElementById("btn-reset-seek").addEventListener("click", resetAllSeekRanges);

  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const folderInput = document.getElementById("folder-input");

  document.getElementById("btn-open-file")?.addEventListener("click", () => fileInput?.click());
  document.getElementById("btn-open-folder")?.addEventListener("click", () => folderInput?.click());

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
      loadFiles(e.dataTransfer.files);
    }
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      loadFiles(e.target.files);
    }
  });
  if (folderInput) {
    folderInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        loadFiles(e.target.files);
      }
    });
  }
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

    // 最終チャンク以外は最後の途切れた行を次回チャンクに持ち越す
    remainder = end < totalBytes ? (lines.pop() || "") : "";

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
        const x = Number(parts[0]);
        const y = Number(parts[1]);
        const z = Number(parts[2]);
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

        pts.push(x, y, z);

        const rIdx = parts.length >= 7 ? 4 : 3;
        if (parts.length >= rIdx + 3) {
          const rv = Number(parts[rIdx]);
          const gv = Number(parts[rIdx + 1]);
          const bv = Number(parts[rIdx + 2]);
          if (Number.isFinite(rv) && Number.isFinite(gv) && Number.isFinite(bv)) {
            const maxC = Math.max(rv, gv, bv);
            const scale = maxC > 255 ? 65535 : (maxC > 1 ? 255 : 1);
            cols.push(
              Math.min(1, Math.max(0, rv / scale)),
              Math.min(1, Math.max(0, gv / scale)),
              Math.min(1, Math.max(0, bv / scale))
            );
          }
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
    const pointsArr = new Float64Array(pts);
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

function updateBadges(normalCount, noiseCount = 0) {
  document.getElementById("point-count-badge").textContent = `通常: ${normalCount.toLocaleString()} 点 | ノイズ: ${noiseCount.toLocaleString()} 点`;
  const lblNormal = document.getElementById("lbl-layer-normal");
  const lblNoise = document.getElementById("lbl-layer-noise");
  if (lblNormal) lblNormal.textContent = `${normalCount.toLocaleString()} 点`;
  if (lblNoise) lblNoise.textContent = `${noiseCount.toLocaleString()} 点`;
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

/**
 * ピル型トースト通知を表示
 */
function showToast(msg, duration = 2600) {
  const toast = document.getElementById("toast-pill");
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add("show");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove("show");
    toastTimer = null;
  }, duration);
}

/**
 * サイドバーの格納/展開を切替
 */
function toggleSidebar() {
  const sidebar = document.getElementById("main-sidebar");
  const toggleBtn = document.getElementById("btn-toggle-sidebar");
  if (!sidebar) return;
  const isCollapsed = sidebar.classList.toggle("collapsed");
  if (toggleBtn) {
    toggleBtn.textContent = isCollapsed ? "▶ パネル" : "◀ パネル";
  }
  setTimeout(onWindowResize, 200);
}

/**
 * ワークフロー切替 (全セクション / レイヤー / 選択・断面 / 自動除去 / 表示)
 */
function setWorkflowTab(tabKey) {
  const tabs = document.querySelectorAll("#sidebar-nav-tabs .flow-tab-btn");
  tabs.forEach((btn) => {
    if (btn.dataset.tab === tabKey) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  const sections = document.querySelectorAll(".panel-section");
  sections.forEach((sec) => {
    if (tabKey === "all") {
      sec.style.display = "";
    } else {
      const match = sec.classList.contains(`sec-${tabKey}`);
      sec.style.display = match ? "" : "none";
    }
  });
}

/**
 * 1クリック操作モード切替 (回転(V) / パン(H) / ブラシ(B) / 矩形(R) / 円形(C) / 多角形(L) / 計測(M))
 */
function setInteractionMode(mode) {
  const selectToggleBtn = document.getElementById("btn-toggle-select");
  const measureToggleBtn = document.getElementById("btn-toggle-measure");
  const shapeSelect = document.getElementById("select-shape-mode");

  if (mode === "orbit" || mode === "pan") {
    isPanMode = (mode === "pan");
    if (controls) {
      controls.enabled = true;
      controls.mouseButtons.LEFT = isPanMode ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
      controls.mouseButtons.RIGHT = isPanMode ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN;
    }
    document.body.style.cursor = isPanMode ? "grab" : "";
    if (isSelectMode) {
      isSelectMode = false;
      if (selectToggleBtn) {
        selectToggleBtn.textContent = "🔲 手動選択モード: OFF";
        selectToggleBtn.classList.remove("btn-active");
      }
      clearSelection();
    }
    if (isMeasureMode) {
      isMeasureMode = false;
      if (measureToggleBtn) {
        measureToggleBtn.textContent = "📏 距離計測: OFF";
        measureToggleBtn.classList.remove("btn-active");
      }
      clearMeasure();
    }
    renderSelectionOverlay();
    if (isPanMode) {
      setStatusMessage("【✋ パン移動モード (H / V)】左ドラッグで平行移動、右ドラッグで回転、ホイールでズーム");
      showToast("✋ パン平行移動モード (左ドラッグで移動)");
    } else {
      setStatusMessage("【🖐 視点回転モード (V)】左ドラッグで回転、右ドラッグでパン、ホイールでズーム");
      showToast("🖐 視点回転モード (V)");
    }
  } else if (mode === "brush" || mode === "rect" || mode === "circle" || mode === "poly") {
    isPanMode = false;
    document.body.style.cursor = "";
    if (controls) {
      controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
      controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
    }
    if (isMeasureMode) {
      isMeasureMode = false;
      if (measureToggleBtn) {
        measureToggleBtn.textContent = "📏 距離計測: OFF";
        measureToggleBtn.classList.remove("btn-active");
      }
      clearMeasure();
    }
    isSelectMode = true;
    controls.enabled = false;
    selectShapeMode = (mode === "poly" ? "polygon" : mode);
    if (shapeSelect) shapeSelect.value = selectShapeMode;
    if (selectToggleBtn) {
      selectToggleBtn.textContent = "🔲 手動選択モード: ON";
      selectToggleBtn.classList.add("btn-active");
    }
    polygonPoints = [];
    brushStrokePoints = [];
    const confirmBtn = document.getElementById("btn-confirm-polygon");
    if (confirmBtn) confirmBtn.disabled = true;
    renderSelectionOverlay();

    if (selectShapeMode === "polygon") {
      setStatusMessage("【⬠ 多角形選択 (L)】頂点を左クリックで追加、Enterまたは右クリックで確定");
    } else if (selectShapeMode === "circle") {
      setStatusMessage("【⭕ 円形選択 (C)】中心から左ドラッグして円形範囲を選択 (Shift追加 / Ctrl除外)");
    } else if (selectShapeMode === "brush") {
      setStatusMessage("【🧹 ブラシ選択/消去 (B)】画面上を左ドラッグでなぞって選択・消去 ([ / ]キーでサイズ変更 / Space+左ドラッグでパン)");
    } else {
      setStatusMessage("【⬜ 矩形選択 (R)】左ドラッグで四角形範囲を選択 (Shift追加 / Ctrl除外)");
    }
  } else if (mode === "measure") {
    isPanMode = false;
    document.body.style.cursor = "";
    if (controls) {
      controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
      controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
    }
    if (isSelectMode) {
      isSelectMode = false;
      controls.enabled = true;
      if (selectToggleBtn) {
        selectToggleBtn.textContent = "🔲 手動選択モード: OFF";
        selectToggleBtn.classList.remove("btn-active");
      }
      clearSelection();
    }
    isMeasureMode = true;
    controls.enabled = true;
    if (measureToggleBtn) {
      measureToggleBtn.textContent = "📏 距離計測: ON";
      measureToggleBtn.classList.add("btn-active");
    }
    clearMeasure();
    renderSelectionOverlay();
    setStatusMessage("【📐 距離計測 (M)】点群上の1点目をクリックしてください");
  }

  syncModeSegments();
}

/**
 * 上部モードセグメントボタン & 左パネルクイック切替ボタンの見た目を現在の状態と同期
 */
function syncModeSegments() {
  const segOrbit = document.getElementById("seg-btn-orbit");
  const segPan = document.getElementById("seg-btn-pan");
  const segBrush = document.getElementById("seg-btn-brush");
  const segRect = document.getElementById("seg-btn-rect");
  const segCircle = document.getElementById("seg-btn-circle");
  const segPoly = document.getElementById("seg-btn-poly");
  const segMeasure = document.getElementById("seg-btn-measure");

  const sideOrbit = document.getElementById("side-btn-orbit");
  const sidePan = document.getElementById("side-btn-pan");
  const sideBrush = document.getElementById("side-btn-brush");
  const sideRect = document.getElementById("side-btn-rect");
  const sideCircle = document.getElementById("side-btn-circle");
  const sidePoly = document.getElementById("side-btn-poly");

  const allSegBtns = [segOrbit, segPan, segBrush, segRect, segCircle, segPoly, segMeasure];
  allSegBtns.forEach((b) => b?.classList.remove("active"));

  const allSideBtns = [sideOrbit, sidePan, sideBrush, sideRect, sideCircle, sidePoly];
  allSideBtns.forEach((b) => b?.classList.remove("btn-active"));

  if (isMeasureMode) {
    segMeasure?.classList.add("active");
  } else if (isSelectMode) {
    if (selectShapeMode === "polygon") {
      segPoly?.classList.add("active");
      sidePoly?.classList.add("btn-active");
    } else if (selectShapeMode === "circle") {
      segCircle?.classList.add("active");
      sideCircle?.classList.add("btn-active");
    } else if (selectShapeMode === "brush") {
      segBrush?.classList.add("active");
      sideBrush?.classList.add("btn-active");
    } else {
      segRect?.classList.add("active");
      sideRect?.classList.add("btn-active");
    }
  } else if (isPanMode) {
    segPan?.classList.add("active");
    sidePan?.classList.add("btn-active");
  } else {
    segOrbit?.classList.add("active");
    sideOrbit?.classList.add("btn-active");
  }
}

/**
 * 選択時フローティング・アクションバーの表示と点数を同期
 */
function updateFloatingSelectionBar(count) {
  const bar = document.getElementById("bottom-overlay-bar");
  const badge = document.getElementById("floating-sel-badge");
  const btnDel = document.getElementById("floating-btn-delete");
  const btnMove = document.getElementById("floating-btn-move");
  const btnRestore = document.getElementById("floating-btn-restore");
  const btnClear = document.getElementById("floating-btn-clear");

  if (!bar) return;
  if (count > 0) {
    bar.style.display = "flex";
    if (badge) badge.textContent = `✨ ${count.toLocaleString()} 点を選択中:`;
    if (btnDel) btnDel.disabled = false;
    if (btnMove) btnMove.disabled = false;
    if (btnRestore) btnRestore.disabled = false;
    if (btnClear) btnClear.disabled = false;
  } else {
    bar.style.display = "none";
  }
}

/**
 * 右クリック文脈メニューの表示
 */
function showContextMenu(clientX, clientY) {
  const menu = document.getElementById("context-menu");
  if (!menu) return;

  updateContextMenuLayersSubmenu();

  menu.style.display = "flex";
  const menuWidth = 220;
  const menuHeight = menu.offsetHeight || 320;
  const posX = Math.min(clientX, window.innerWidth - menuWidth - 10);
  const posY = Math.min(clientY, window.innerHeight - menuHeight - 10);

  menu.style.left = `${Math.max(10, posX)}px`;
  menu.style.top = `${Math.max(10, posY)}px`;
}

/**
 * 右クリック文脈メニューを非表示
 */
function hideContextMenu() {
  const menu = document.getElementById("context-menu");
  if (menu) menu.style.display = "none";
}

/**
 * 右クリックメニュー内のレイヤー移動サブメニューを更新
 */
function updateContextMenuLayersSubmenu() {
  const submenu = document.getElementById("ctx-submenu-layers");
  if (!submenu) return;
  submenu.innerHTML = "";

  for (const [lidStr, lcfg] of Object.entries(layersConfig)) {
    const lid = Number(lidStr);
    const item = document.createElement("div");
    item.className = "ctx-item";
    // レイヤー名・色はユーザー入力や読み込んだプロジェクトJSON由来のため innerHTML には入れない（XSS防止）
    const dot = document.createElement("span");
    dot.style.cssText = "display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;";
    dot.style.background = String(lcfg.color || "#888");
    item.appendChild(dot);
    item.appendChild(document.createTextNode(String(lcfg.name ?? "")));
    item.addEventListener("click", () => {
      hideContextMenu();
      const selTarget = document.getElementById("select-target-layer");
      if (selTarget) selTarget.value = String(lid);
      moveSelectedToTargetLayer();
    });
    submenu.appendChild(item);
  }
}

/**
 * 複数ファイルまたはフォルダ内の点群を一括読み込み
 */
async function loadFiles(fileList) {
  if (!fileList || fileList.length === 0) return;
  const validFiles = Array.from(fileList).filter((f) => {
    const ext = f.name.split(".").pop().toLowerCase();
    return ["ply", "xyz", "pts", "csv", "txt"].includes(ext);
  });

  if (validFiles.length === 0) {
    alert("読み込み可能な点群ファイル (PLY, XYZ, PTS, CSV) が見つかりませんでした。");
    return;
  }

  if (validFiles.length === 1) {
    await loadFile(validFiles[0]);
    return;
  }

  await loadMultipleFiles(validFiles);
}

/**
 * 複数ファイルの点群データを順次結合して統合読み込み
 */
async function loadMultipleFiles(files) {
  showLoading(`複数ファイル一括読み込み中 (全 ${files.length} ファイル)...`);
  const budgetVal = parseInt(document.getElementById("select-point-budget")?.value || "1000000", 10);
  const targetBudget = budgetVal > 0 ? budgetVal : 50000000;
  const perFileBudget = Math.max(10000, Math.floor(targetBudget / files.length));

  const allPts = [];
  const allCols = [];
  let hasAnyColor = false;

  for (let idx = 0; idx < files.length; idx++) {
    const f = files[idx];
    showLoadingProgress(idx, files.length, allPts.length / 3, `ファイル読込中 (${idx + 1}/${files.length}): ${f.name}`);
    await new Promise((resolve) => setTimeout(resolve, 0));

    try {
      const text = await f.text();
      const { points, colors } = parseTextPointCloud(text);
      const fileTotal = points.length / 3;
      const step = Math.max(1, Math.floor(fileTotal / perFileBudget));
      const hasFileColor = Boolean(colors && colors.length === points.length);
      if (hasFileColor) hasAnyColor = true;

      for (let i = 0; i < fileTotal; i += step) {
        allPts.push(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
        if (hasFileColor) {
          allCols.push(colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]);
        } else {
          allCols.push(0.3, 0.8, 0.95);
        }
      }
    } catch (err) {
      console.warn(`ファイル読み込みエラー: ${f.name}`, err);
    }
  }

  hideLoading();

  if (allPts.length > 0) {
    const pointsArr = new Float64Array(allPts);
    const colorsArr = hasAnyColor && allCols.length === allPts.length ? new Float32Array(allCols) : null;
    originalPoints = pointsArr.slice();
    undoStack.length = 0;
    redoStack.length = 0;
    updateUndoRedoUI();
    setPointCloud(pointsArr, colorsArr);
    setStatusMessage(`フォルダ/複数ファイル (${files.length} 個) から ${(allPts.length / 3).toLocaleString()} 点を統合読み込みしました`);
    showToast(`📂 ${files.length} 個のファイルを統合読み込みしました`);
  } else {
    alert("点群データを解析できませんでした。");
  }
}

/**
 * 即時削除モード（囲んだら即削除）のトグル切替
 */
function toggleInstantDeleteMode(forceVal) {
  instantDeleteMode = typeof forceVal === "boolean" ? forceVal : !instantDeleteMode;
  const btnTop = document.getElementById("btn-instant-del");
  const btnSide = document.getElementById("chk-instant-del");
  if (btnTop) btnTop.classList.toggle("btn-active", instantDeleteMode);
  if (btnSide) {
    btnSide.classList.toggle("btn-active", instantDeleteMode);
    btnSide.textContent = instantDeleteMode
      ? "⚡ 囲んだら即削除: ON (1ステップ消去中)"
      : "⚡ 囲んだら即削除 (Deleteキー不要の1ステップ消去)";
  }
  renderSelectionOverlay();
  if (instantDeleteMode) {
    showToast("⚡ 即時削除モード ON: 囲んだ瞬間にノイズへ即消去します");
  } else {
    showToast("⚡ 即時削除モード OFF: 通常選択に戻しました");
  }
}

/**
 * 🎯 ノイズ処理専用モード（高コントラスト背景・点サイズ強調・即時削除ON・ブラシ起動）のトグル切替
 */
function toggleCleanFocusMode(forceVal) {
  cleanFocusMode = typeof forceVal === "boolean" ? forceVal : !cleanFocusMode;
  const btnTop = document.getElementById("btn-clean-mode");
  const btnSide = document.getElementById("btn-clean-mode-side");
  if (btnTop) btnTop.classList.toggle("btn-active", cleanFocusMode);
  if (btnSide) {
    btnSide.classList.toggle("btn-active", cleanFocusMode);
    btnSide.textContent = cleanFocusMode
      ? "🎯 ノイズ処理専用モード: ON (高コントラスト・即消去)"
      : "🎯 ノイズ処理専用モード (N)";
  }

  const sizeSlider = document.getElementById("slider-point-size");
  if (cleanFocusMode) {
    savedPreCleanState = {
      bgColor: appSettings.bgColor,
      pointSize: sizeSlider ? parseFloat(sizeSlider.value) : 1.2,
      instantDelete: instantDeleteMode,
    };
    if (scene) scene.background = new THREE.Color("#04060b");
    if (sizeSlider) {
      const boostSize = Math.max(2.2, savedPreCleanState.pointSize);
      sizeSlider.value = String(boostSize);
      document.getElementById("label-point-size").textContent = `${boostSize.toFixed(1)} px`;
      if (pointCloud) {
        const shape = document.getElementById("select-point-shape")?.value || "circle";
        applyPointMaterialProperties(pointCloud.material, shape, boostSize);
      }
    }
    toggleInstantDeleteMode(true);
    setInteractionMode("brush");
    setStatusMessage("🎯【ノイズ処理専用モード ON】ブラシや選択枠でなぞった箇所を即座に消去します (Alt+左ドラッグで回転)");
    showToast("🎯 ノイズ処理専用モード ON (ブラシ即消去・高コントラスト表示)");
  } else {
    if (savedPreCleanState) {
      if (scene) scene.background = new THREE.Color(savedPreCleanState.bgColor || "#0b0f17");
      if (sizeSlider && savedPreCleanState.pointSize) {
        sizeSlider.value = String(savedPreCleanState.pointSize);
        document.getElementById("label-point-size").textContent = `${savedPreCleanState.pointSize.toFixed(1)} px`;
        if (pointCloud) {
          const shape = document.getElementById("select-point-shape")?.value || "circle";
          applyPointMaterialProperties(pointCloud.material, shape, savedPreCleanState.pointSize);
        }
      }
      toggleInstantDeleteMode(Boolean(savedPreCleanState.instantDelete));
    } else {
      toggleInstantDeleteMode(false);
    }
    setStatusMessage("ノイズ処理専用モードを終了し、通常表示に戻しました");
  }
  renderSelectionOverlay();
}

/**
 * ワンクリック・プリセット適用 (大まか / 標準 / 細かい)
 */
function applyFilterPreset(presetKey) {
  const kSlider = document.getElementById("slider-sor-k");
  const kLabel = document.getElementById("label-sor-k");
  const stdSlider = document.getElementById("slider-sor-std");
  const stdLabel = document.getElementById("label-sor-std");
  if (!kSlider || !stdSlider) return;

  if (presetKey === "rough") {
    kSlider.value = "15";
    stdSlider.value = "2.8";
    if (kLabel) kLabel.textContent = "15";
    if (stdLabel) stdLabel.textContent = "2.8";
    showToast("🟢 大まか(粗取り)プリセットをセットしました (k=15, σ=2.8)");
    setStatusMessage("🟢【大まか(粗取り)プリセット】遠くに浮いた大まかな飛散ノイズの除去に最適です");
  } else if (presetKey === "fine") {
    kSlider.value = "35";
    stdSlider.value = "1.1";
    if (kLabel) kLabel.textContent = "35";
    if (stdLabel) stdLabel.textContent = "1.1";
    showToast("🟠 細かい(精密仕上げ)プリセットをセットしました (k=35, σ=1.1)");
    setStatusMessage("🟠【細かい(精密仕上げ)プリセット】輪郭付近の微細ノイズやヒゲの精密除去に最適です");
  } else {
    kSlider.value = "20";
    stdSlider.value = "2.0";
    if (kLabel) kLabel.textContent = "20";
    if (stdLabel) stdLabel.textContent = "2.0";
    showToast("🟡 標準バランスプリセットをセットしました (k=20, σ=2.0)");
    setStatusMessage("🟡【標準バランスプリセット】一般的な点群ノイズ除去の標準設定です");
  }
}

/**
 * 自動除去用のノイズレイヤーを取得または新規作成
 */
function ensureAutoNoiseLayer(layerName, colorHex = "#f97316") {
  for (const [k, v] of Object.entries(layersConfig)) {
    if (v.name === layerName) return Number(k);
  }
  const newLid = Object.keys(layersConfig).length > 0
    ? Math.max(...Object.keys(layersConfig).map(Number)) + 1
    : 10;
  layersConfig[newLid] = {
    id: newLid,
    name: layerName,
    visible: false,
    locked: false,
    color: colorHex,
    useLayerColor: true,
    isNoise: true,
  };
  return newLid;
}

/**
 * 🪄 マジックワンド：現在選択されている点から3D空間で繋がっている塊（クラスタ）を一括選択 (Gキー)
 */
function magicWandClusterSelect() {
  if (!currentPoints || !layerMask) return;
  if (selectedIndices.size === 0) {
    showToast("💡 先にノイズや対象物の端を少しだけ選択してから「🪄 連結塊選択(G)」を押してください");
    setStatusMessage("先に点群の一部を選択してから「🪄 連結塊選択 (G)」を実行してください");
    return;
  }

  const total = currentPoints.length / 3;
  // 点群スケールから適切な3D連結許容距離を自動算出
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
  let sumSeedZ = 0;
  for (const idx of selectedIndices) {
    sumSeedZ += currentPoints[idx * 3 + 2];
  }
  const seedMeanZ = sumSeedZ / selectedIndices.size;

  const step = Math.max(1, Math.floor(total / 2000));
  for (let i = 0; i < total; i += step) {
    if (!isPointVisible(i)) continue;
    const x = currentPoints[i * 3], y = currentPoints[i * 3 + 1], z = currentPoints[i * 3 + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const diag = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) || 10.0;
  const tol = Math.max(0.05, diag * 0.018);
  const invCell = 1.0 / tol;
  const zFloorLimit = seedMeanZ - tol * 2.5;

  // 3D空間ハッシュグリッド構築
  const grid = new Map();
  for (let i = 0; i < total; i++) {
    if (!isPointVisible(i) || isLayerLocked(layerMask[i])) continue;
    const z = currentPoints[i * 3 + 2];
    if (z < zFloorLimit) continue;
    const gx = Math.floor(currentPoints[i * 3] * invCell);
    const gy = Math.floor(currentPoints[i * 3 + 1] * invCell);
    const gz = Math.floor(z * invCell);
    const key = `${gx},${gy},${gz}`;
    let bucket = grid.get(key);
    if (!bucket) {
      bucket = [];
      grid.set(key, bucket);
    }
    bucket.push(i);
  }

  const tolSq = tol * tol;
  const visited = new Set(selectedIndices);
  const queue = Array.from(selectedIndices);
  const maxGrow = Math.max(selectedIndices.size * 50, Math.min(80000, Math.floor(total * 0.25)));
  let head = 0;

  while (head < queue.length && visited.size < maxGrow) {
    const curr = queue[head++];
    const cx = currentPoints[curr * 3];
    const cy = currentPoints[curr * 3 + 1];
    const cz = currentPoints[curr * 3 + 2];
    const gx = Math.floor(cx * invCell);
    const gy = Math.floor(cy * invCell);
    const gz = Math.floor(cz * invCell);

    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const bucket = grid.get(`${gx + dx},${gy + dy},${gz + dz}`);
          if (!bucket) continue;
          for (let k = 0; k < bucket.length; k++) {
            const nb = bucket[k];
            if (visited.has(nb)) continue;
            const ddx = currentPoints[nb * 3] - cx;
            const ddy = currentPoints[nb * 3 + 1] - cy;
            const ddz = currentPoints[nb * 3 + 2] - cz;
            if (ddx * ddx + ddy * ddy + ddz * ddz <= tolSq) {
              visited.add(nb);
              queue.push(nb);
              if (visited.size >= maxGrow) break;
            }
          }
        }
      }
    }
  }

  const beforeCount = selectedIndices.size;
  selectedIndices = visited;
  if (instantDeleteMode) {
    deleteSelectedPoints();
    return;
  }
  updatePointCloudDisplay();
  setStatusMessage(`🪄 連結塊選択: ${beforeCount.toLocaleString()} 点 → 繋がっている ${selectedIndices.size.toLocaleString()} 点を一括選択しました (Deleteで消去)`);
  showToast(`🪄 連結する塊 ${selectedIndices.size.toLocaleString()} 点を一括選択しました`);
}

/**
 * 📦 選択範囲内のみ局所精密ノイズ除去 (周囲の細部を守りつつ選択箇所の細かいノイズだけ除去)
 */
function localFineCleanSelected() {
  if (!currentPoints || !layerMask) return;
  if (selectedIndices.size < 8) {
    showToast("💡 先にノイズが気になる箇所を囲んで選択してから「📦 選択内のみ除去」を押してください");
    setStatusMessage("選択範囲が少なすぎます（8点以上を選択してから実行してください）");
    return;
  }

  const selArr = Array.from(selectedIndices).filter((i) => isPointUnlockedNormal(i));
  if (selArr.length < 8) {
    setStatusMessage("選択範囲内に編集可能な通常レイヤー点がありません");
    return;
  }

  const meanDists = new Float32Array(selArr.length);
  let sumMean = 0;
  const kNeighbors = Math.min(16, selArr.length - 1);
  const sampleStep = Math.max(1, Math.floor(selArr.length / 400));

  for (let idx = 0; idx < selArr.length; idx++) {
    const i = selArr[idx];
    const xi = currentPoints[i * 3], yi = currentPoints[i * 3 + 1], zi = currentPoints[i * 3 + 2];
    const dists = [];
    for (let jIdx = 0; jIdx < selArr.length; jIdx += sampleStep) {
      if (idx === jIdx) continue;
      const j = selArr[jIdx];
      const dx = xi - currentPoints[j * 3];
      const dy = yi - currentPoints[j * 3 + 1];
      const dz = zi - currentPoints[j * 3 + 2];
      dists.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
    }
    dists.sort((a, b) => a - b);
    const kUse = Math.min(kNeighbors, dists.length);
    let dSum = 0;
    for (let k = 0; k < kUse; k++) dSum += dists[k];
    const avg = kUse > 0 ? dSum / kUse : 0;
    meanDists[idx] = avg;
    sumMean += avg;
  }

  const mean = sumMean / selArr.length;
  let varSum = 0;
  for (let idx = 0; idx < selArr.length; idx++) {
    varSum += (meanDists[idx] - mean) ** 2;
  }
  const std = Math.sqrt(varSum / selArr.length);
  const threshold = mean + 1.05 * std;

  const noiseHits = [];
  for (let idx = 0; idx < selArr.length; idx++) {
    if (meanDists[idx] > threshold) {
      noiseHits.push(selArr[idx]);
    }
  }

  if (noiseHits.length === 0) {
    showToast("✨ 選択範囲内に突出した局所ノイズは検出されませんでした");
    return;
  }

  pushUndoState();
  const autoLid = ensureAutoNoiseLayer("局所精密除去ノイズ", "#f43f5e");
  for (const idx of noiseHits) {
    layerMask[idx] = autoLid;
  }
  selectedIndices.clear();
  rebuildBaseColors();
  updatePointCloudDisplay();
  updateUndoRedoUI();
  setStatusMessage(`📦 選択範囲内 (${selArr.length.toLocaleString()} 点) から微細ノイズ ${noiseHits.length.toLocaleString()} 点を精密除去しました`);
  showToast(`📦 選択内から ${noiseHits.length.toLocaleString()} 点の細かいノイズを除去しました`);
}

/**
 * 🏝️ 大まかなノイズ処理：メイン塊のみ残す（最大連結クラスタ以外の一括浮遊物除去）
 */
function keepMainClusterOnly() {
  if (!currentPoints || !layerMask) return;
  const total = currentPoints.length / 3;
  const normalIndices = [];
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;

  for (let i = 0; i < total; i++) {
    if (isPointUnlockedNormal(i)) {
      normalIndices.push(i);
      const x = currentPoints[i * 3], y = currentPoints[i * 3 + 1], z = currentPoints[i * 3 + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
  }
  if (normalIndices.length < 10) return;

  const diag = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) || 10.0;
  const cellSize = Math.max(0.08, diag * 0.035);
  const invCell = 1.0 / cellSize;

  // ボクセル単位で連結成分分解を実行（高速・省メモリ）
  const voxelMap = new Map();
  for (let idx = 0; idx < normalIndices.length; idx++) {
    const i = normalIndices[idx];
    const gx = Math.floor(currentPoints[i * 3] * invCell);
    const gy = Math.floor(currentPoints[i * 3 + 1] * invCell);
    const gz = Math.floor(currentPoints[i * 3 + 2] * invCell);
    const key = `${gx},${gy},${gz}`;
    let list = voxelMap.get(key);
    if (!list) {
      list = [];
      voxelMap.set(key, list);
    }
    list.push(i);
  }

  const visitedVoxels = new Set();
  let bestClusterPoints = [];

  for (const [startKey, startPts] of voxelMap.entries()) {
    if (visitedVoxels.has(startKey)) continue;
    visitedVoxels.add(startKey);
    const queue = [startKey];
    const clusterPts = [];
    let head = 0;

    while (head < queue.length) {
      const currKey = queue[head++];
      const pts = voxelMap.get(currKey);
      if (pts) {
        for (let k = 0; k < pts.length; k++) clusterPts.push(pts[k]);
      }
      const [cx, cy, cz] = currKey.split(",").map(Number);
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dz = -1; dz <= 1; dz++) {
            if (dx === 0 && dy === 0 && dz === 0) continue;
            const nbKey = `${cx + dx},${cy + dy},${cz + dz}`;
            if (!visitedVoxels.has(nbKey) && voxelMap.has(nbKey)) {
              visitedVoxels.add(nbKey);
              queue.push(nbKey);
            }
          }
        }
      }
    }

    if (clusterPts.length > bestClusterPoints.length) {
      bestClusterPoints = clusterPts;
    }
  }

  const keepSet = new Set(bestClusterPoints);
  const floatingNoise = normalIndices.filter((i) => !keepSet.has(i));

  if (floatingNoise.length === 0) {
    showToast("🏝️ 分離した浮遊塊ノイズは見つかりませんでした");
    setStatusMessage("すべての有効点がメインクラスタに連結しています");
    return;
  }

  pushUndoState();
  const autoLid = ensureAutoNoiseLayer("自動除去: 浮遊塊(メイン外)", "#a855f7");
  for (const i of floatingNoise) {
    layerMask[i] = autoLid;
  }
  selectedIndices.clear();
  rebuildBaseColors();
  updatePointCloudDisplay();
  updateUndoRedoUI();
  setStatusMessage(`🏝️ メイン塊 (${bestClusterPoints.length.toLocaleString()} 点) のみを残し、周囲の浮遊ノイズ ${floatingNoise.length.toLocaleString()} 点を一括除去しました`);
  showToast(`🏝️ 浮遊塊ノイズ ${floatingNoise.length.toLocaleString()} 点を一括除去しました`);
}

/**
 * 🪶 細かいノイズ処理：エッジ飛散・ヒゲ（ベールノイズ）をワンクリック精密除去
 */
function quickEdgeVeilClean() {
  if (!currentPoints || !layerMask) return;
  showLoading("🪶 エッジ飛散・ヒゲ(微細ノイズ)を精密解析中...");

  setTimeout(() => {
    const total = currentPoints.length / 3;
    const normalIndices = [];
    for (let i = 0; i < total; i++) {
      if (isPointUnlockedNormal(i)) normalIndices.push(i);
    }
    if (normalIndices.length < 15) {
      hideLoading();
      return;
    }

    const meanDists = new Float32Array(normalIndices.length);
    let sumMean = 0;
    const step = Math.max(1, Math.floor(normalIndices.length / 450));

    for (let idx = 0; idx < normalIndices.length; idx++) {
      const i = normalIndices[idx];
      const xi = currentPoints[i * 3], yi = currentPoints[i * 3 + 1], zi = currentPoints[i * 3 + 2];
      let d1 = Infinity, d2 = Infinity, d3 = Infinity, d4 = Infinity;

      for (let jIdx = 0; jIdx < normalIndices.length; jIdx += step) {
        const j = normalIndices[jIdx];
        if (i === j) continue;
        const dx = xi - currentPoints[j * 3];
        const dy = yi - currentPoints[j * 3 + 1];
        const dz = zi - currentPoints[j * 3 + 2];
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (d < d1) { d4 = d3; d3 = d2; d2 = d1; d1 = d; }
        else if (d < d2) { d4 = d3; d3 = d2; d2 = d; }
        else if (d < d3) { d4 = d3; d3 = d; }
        else if (d < d4) { d4 = d; }
      }
      // 最近傍4点の平均距離と局所延伸比(d4/d1)を組み合わせてエッジ・ヒゲを検出
      const avgNear = (d1 + d2 + d3 + d4) * 0.25;
      const stretchRatio = d1 > 1e-6 ? Math.min(3.0, (d4 / d1) * 0.35) : 1.0;
      const score = avgNear * (1.0 + stretchRatio * 0.25);
      meanDists[idx] = score;
      sumMean += score;
    }

    const mean = sumMean / normalIndices.length;
    let varSum = 0;
    for (let idx = 0; idx < normalIndices.length; idx++) {
      varSum += (meanDists[idx] - mean) ** 2;
    }
    const std = Math.sqrt(varSum / normalIndices.length);
    const threshold = mean + 1.25 * std;

    const veilHits = [];
    for (let idx = 0; idx < normalIndices.length; idx++) {
      if (meanDists[idx] > threshold) {
        veilHits.push(normalIndices[idx]);
      }
    }

    hideLoading();
    if (veilHits.length === 0) {
      showToast("🪶 エッジ飛散・ヒゲノイズは検出されませんでした");
      return;
    }

    pushUndoState();
    const autoLid = ensureAutoNoiseLayer("自動除去: エッジ・ヒゲ", "#ec4899");
    for (const i of veilHits) {
      layerMask[i] = autoLid;
    }
    selectedIndices.clear();
    rebuildBaseColors();
    updatePointCloudDisplay();
    updateUndoRedoUI();
    setStatusMessage(`🪶 エッジ飛散・ヒゲ除去: 輪郭部の微細ノイズ ${veilHits.length.toLocaleString()} 点を「自動除去: エッジ・ヒゲ」へ分離しました`);
    showToast(`🪶 エッジ・ヒゲノイズ ${veilHits.length.toLocaleString()} 点を精密除去しました`);
  }, 40);
}

