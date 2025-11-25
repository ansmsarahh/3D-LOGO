
/* -----------------------------
   Shaders (vertex + fragment)
   ----------------------------- */
const VS_SOURCE = `
attribute vec3 aPos;
attribute vec3 aNormal;
attribute vec3 aColor;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat4 uNormalMatrix;

varying vec3 vNormal;
varying vec3 vWorldPos;
varying vec3 vColor;

void main() {
  vec4 world = uModel * vec4(aPos, 1.0);
  vWorldPos = world.xyz;
  vNormal = mat3(uNormalMatrix) * aNormal;
  vColor = aColor;
  gl_Position = uProj * uView * world;
}
`;

const FS_SOURCE = `
precision mediump float;
varying vec3 vNormal;
varying vec3 vWorldPos;
varying vec3 vColor;

uniform vec3 uLightDir;    // directional light
uniform vec3 uViewPos;
uniform float uAmbient;
uniform float uSpecStrength;
uniform float uShininess;

void main() {
  vec3 N = normalize(vNormal);
  vec3 L = normalize(-uLightDir);
  vec3 V = normalize(uViewPos - vWorldPos);

  float diff = max(dot(N, L), 0.0);
  vec3 R = reflect(-L, N);
  float spec = pow(max(dot(R, V), 0.0), uShininess);

  // rim-ish accent for brightness near grazing angles
  float rim = pow(1.0 - max(dot(V,N), 0.0), 2.5) * 0.15;

  vec3 color = vColor * (uAmbient + diff * 0.95) + spec * uSpecStrength * vec3(1.0) + rim * vec3(1.0);
  // simple gamma correction
  color = pow(color, vec3(1.0/1.9));
  gl_FragColor = vec4(color, 1.0);
}
`;

/* -----------------------------
   Utilities: matrix helpers
   (small set sufficient for this demo)
   ----------------------------- */
function identity() { return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; }
function multiply(a,b){
  const out = new Array(16);
  for (let i=0;i<4;i++){
    for (let j=0;j<4;j++){
      let s=0;
      for (let k=0;k<4;k++) s += a[i*4+k]*b[k*4+j];
      out[i*4+j]=s;
    }
  }
  return out;
}
function translation(tx,ty,tz){ const m=identity(); m[12]=tx; m[13]=ty; m[14]=tz; return m; }
function scaleMat(sx,sy,sz){ const m=identity(); m[0]=sx; m[5]=sy; m[10]=sz; return m; }
function rotateY(theta){ const c=Math.cos(theta), s=Math.sin(theta); return [ c,0,s,0, 0,1,0,0, -s,0,c,0, 0,0,0,1 ]; }
function rotateX(theta){ const c=Math.cos(theta), s=Math.sin(theta); return [1,0,0,0, 0,c,-s,0, 0,s,c,0, 0,0,0,1 ]; }
function perspective(fovy, aspect, near, far){
  const f = 1.0/Math.tan(fovy/2), nf = 1/(near - far);
  const out = new Array(16).fill(0);
  out[0]=f/aspect; out[5]=f; out[10]=(far+near)*nf; out[11]=-1; out[14]=2*far*near*nf;
  return out;
}
function lookAt(eye, center, up){
  const z0=eye[0]-center[0], z1=eye[1]-center[1], z2=eye[2]-center[2];
  let len = Math.hypot(z0,z1,z2); if(len===0) { z2=1; len=1; }
  const zx=z0/len, zy=z1/len, zz=z2/len;
  const xx = up[1]*zz - up[2]*zy;
  const xy = up[2]*zx - up[0]*zz;
  const xz = up[0]*zy - up[1]*zx;
  let l = Math.hypot(xx,xy,xz); if(l===0) l=1;
  const ux=xx/l, uy=xy/l, uz=xz/l;
  const vx = zy*uz - zz*uy, vy = zz*ux - zx*uz, vz = zx*uy - zy*ux;
  const out = identity();
  out[0]=ux; out[4]=vx; out[8]=zx;
  out[1]=uy; out[5]=vy; out[9]=zy;
  out[2]=uz; out[6]=vz; out[10]=zz;
  out[12]=-(ux*eye[0]+uy*eye[1]+uz*eye[2]);
  out[13]=-(vx*eye[0]+vy*eye[1]+vz*eye[2]);
  out[14]=-(zx*eye[0]+zy*eye[1]+zz*eye[2]);
  return out;
}
function inverseTransposeMat4(m){
  // compute inverse-transpose of upper-left 3x3, return 4x4 with rest identity
  const a00 = m[0], a01=m[4], a02=m[8];
  const a10 = m[1], a11=m[5], a12=m[9];
  const a20 = m[2], a21=m[6], a22=m[10];
  const det = a00*(a11*a22 - a12*a21) - a01*(a10*a22 - a12*a20) + a02*(a10*a21 - a11*a20);
  if (Math.abs(det) < 1e-8) return identity();
  const invDet = 1/det;
  const b00 = (a11*a22 - a12*a21)*invDet;
  const b01 = (a02*a21 - a01*a22)*invDet;
  const b02 = (a01*a12 - a02*a11)*invDet;
  const b10 = (a12*a20 - a10*a22)*invDet;
  const b11 = (a00*a22 - a02*a20)*invDet;
  const b12 = (a02*a10 - a00*a12)*invDet;
  const b20 = (a10*a21 - a11*a20)*invDet;
  const b21 = (a01*a20 - a00*a21)*invDet;
  const b22 = (a00*a11 - a01*a10)*invDet;
  return [ b00, b10, b20, 0, b01, b11, b21, 0, b02, b12, b22, 0, 0,0,0,1 ];
}

/* -----------------------------
   WebGL setup
   ----------------------------- */
const canvas = document.getElementById('glcanvas');
const gl = canvas.getContext('webgl', {antialias:true});
if(!gl){ alert('WebGL not available'); throw new Error('WebGL not available'); }

let program = null;
function createProgram() {
  const vs = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vs, VS_SOURCE); gl.compileShader(vs);
  if(!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(vs));
  const fs = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fs, FS_SOURCE); gl.compileShader(fs);
  if(!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(fs));
  program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if(!gl.getProgramParameter(program, gl.LINK_STATUS)) console.error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
}
createProgram();

/* attribute + uniform locations */
const loc = {
  aPos: gl.getAttribLocation(program, 'aPos'),
  aNormal: gl.getAttribLocation(program, 'aNormal'),
  aColor: gl.getAttribLocation(program, 'aColor'),
  uModel: gl.getUniformLocation(program, 'uModel'),
  uView: gl.getUniformLocation(program, 'uView'),
  uProj: gl.getUniformLocation(program, 'uProj'),
  uNormalMatrix: gl.getUniformLocation(program, 'uNormalMatrix'),
  uLightDir: gl.getUniformLocation(program, 'uLightDir'),
  uViewPos: gl.getUniformLocation(program, 'uViewPos'),
  uAmbient: gl.getUniformLocation(program, 'uAmbient'),
  uSpecStrength: gl.getUniformLocation(program, 'uSpecStrength'),
  uShininess: gl.getUniformLocation(program, 'uShininess')
};

/* -----------------------------
   Buffers
   ----------------------------- */
let vboPos = gl.createBuffer();
let vboNormal = gl.createBuffer();
let vboColor = gl.createBuffer();
let ibo = gl.createBuffer();
let indexCount = 0;

/* -----------------------------
   UI Elements
   ----------------------------- */
const elDepth = document.getElementById('depth');
const elSpeed = document.getElementById('speed');
const elColor = document.getElementById('color');
const elMode = document.getElementById('mode');
const btnStart = document.getElementById('startBtn');
const btnStop = document.getElementById('stopBtn');
const btnReset = document.getElementById('resetBtn');
const depthVal = document.getElementById('depthVal');
const speedVal = document.getElementById('speedVal');

depthVal.textContent = elDepth.value;
speedVal.textContent = elSpeed.value;

let params = {
  depth: parseFloat(elDepth.value),
  speed: parseFloat(elSpeed.value),
  color: hexToRgb(elColor.value),
  mode: elMode.value
};

/* -----------------------------
   Geometry creation (A, B, L)
   We'll use createPrism for rectangular/extruded blocks and assemble letters
   ----------------------------- */

function createPrism(frontQuad, depth, colorVec) {
  // frontQuad: array of 4 2D points [x,y] (CCW)
  const d = depth;
  const frontZ = +d/2;
  const backZ = -d/2;
  const verts = [];
  for (let i=0;i<4;i++) verts.push([frontQuad[i][0], frontQuad[i][1], frontZ]);
  for (let i=0;i<4;i++) verts.push([frontQuad[i][0], frontQuad[i][1], backZ]);

  const positions = [];
  const normals = [];
  const colors = [];
  const indices = [];
  let idx = 0;

  function push(v, n){
    positions.push(v[0], v[1], v[2]);
    normals.push(n[0], n[1], n[2]);
    colors.push(colorVec[0], colorVec[1], colorVec[2]);
    return idx++;
  }

  // front face (0..3)
  const nf = [0,0,1];
  const i0 = push(verts[0], nf);
  const i1 = push(verts[1], nf);
  const i2 = push(verts[2], nf);
  const i3 = push(verts[3], nf);
  indices.push(i0,i1,i2, i0,i2,i3);

  // back face (4..7) - reversed winding
  const nb = [0,0,-1];
  const b0 = push(verts[4], nb);
  const b1 = push(verts[5], nb);
  const b2 = push(verts[6], nb);
  const b3 = push(verts[7], nb);
  indices.push(b0,b2,b1, b0,b3,b2);

  // sides
  for (let i=0;i<4;i++){
    const inext = (i+1)%4;
    const v0 = verts[i], v1 = verts[inext], v0b = verts[i+4], v1b = verts[inext+4];
    // compute normal using cross product of two edges
    const ex = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
    const ey = [v0b[0]-v0[0], v0b[1]-v0[1], v0b[2]-v0[2]];
    let nx = ex[1]*ey[2]-ex[2]*ey[1];
    let ny = ex[2]*ey[0]-ex[0]*ey[2];
    let nz = ex[0]*ey[1]-ex[1]*ey[0];
    const len = Math.hypot(nx,ny,nz) || 1;
    nx/=len; ny/=len; nz/=len;
    const a = push(v0, [nx,ny,nz]);
    const b = push(v1, [nx,ny,nz]);
    const c = push(v1b, [nx,ny,nz]);
    const dIdx = push(v0b, [nx,ny,nz]);
    indices.push(a,b,c, a,c,dIdx);
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    colors: new Float32Array(colors),
    indices: new Uint16Array(indices)
  };
}

function createGeometry(depth) {
  // Build A, B, L using rectangular blocks (as before), placed left-to-right
  const col = params.color;
  const allPos = [];
  const allNorm = [];
  const allCol = [];
  const allIdx = [];
  let idxOff = 0;

  function append(mesh){
    for(let i=0;i<mesh.positions.length;i++) allPos.push(mesh.positions[i]);
    for(let i=0;i<mesh.normals.length;i++) allNorm.push(mesh.normals[i]);
    for(let i=0;i<mesh.colors.length;i++) allCol.push(mesh.colors[i]);
    for(let i=0;i<mesh.indices.length;i++) allIdx.push(mesh.indices[i] + idxOff);
    idxOff += mesh.positions.length/3;
  }

  const s = 0.95;

  // A: left leg, right leg, crossbar
  const aLeft = [[-2.2*s,-1.0*s],[ -1.8*s,1.0*s],[ -1.4*s,1.0*s],[ -1.9*s,-1.0*s ]];
  const aRight= [[ -1.4*s,1.0*s],[ -1.0*s,-1.0*s],[ -0.6*s,-1.0*s],[ -1.0*s,1.0*s ]];
  const aBar  = [[ -1.7*s,0.05*s],[ -1.0*s,0.05*s],[ -1.0*s,-0.05*s],[ -1.7*s,-0.05*s ]];

  append(createPrism(aLeft, depth, col));
  append(createPrism(aRight, depth, col));
  append(createPrism(aBar, depth, col));

  // B: spine + upper bulge + lower bulge (approx)
  const bSpine = [[-0.2*s,-1.0*s],[0.0*s,-1.0*s],[0.0*s,1.0*s],[ -0.2*s,1.0*s]];
  const bUpper = [[0.0*s,0.25*s],[0.7*s,0.25*s],[0.7*s,0.95*s],[0.0*s,0.95*s]];
  const bLower = [[0.0*s,-0.95*s],[0.7*s,-0.95*s],[0.7*s,-0.25*s],[0.0*s,-0.25*s]];

  append(createPrism(bSpine, depth, col));
  append(createPrism(bUpper, depth, col));
  append(createPrism(bLower, depth, col));

  // L: vertical + bottom bar
  const lVert = [[1.0*s,-1.0*s],[1.2*s,-1.0*s],[1.2*s,1.0*s],[1.0*s,1.0*s]];
  const lBottom = [[1.2*s,-1.0*s],[2.0*s,-1.0*s],[2.0*s,-0.7*s],[1.2*s,-0.7*s]];

  append(createPrism(lVert, depth, col));
  append(createPrism(lBottom, depth, col));

  // upload data to GL buffers
  gl.bindBuffer(gl.ARRAY_BUFFER, vboPos);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(allPos), gl.STATIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, vboNormal);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(allNorm), gl.STATIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, vboColor);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(allCol), gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  const idxArr = new Uint16Array(allIdx);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idxArr, gl.STATIC_DRAW);
  indexCount = idxArr.length;
}

/* Only update color buffer (no geometry rebuild) */
function updateColorBuffer(colorVec) {
  // Recreate color array same length as previously uploaded positions
  // get number of vertices by querying position buffer size
  gl.bindBuffer(gl.ARRAY_BUFFER, vboPos);
  const posSize = gl.getBufferParameter(gl.ARRAY_BUFFER, gl.BUFFER_SIZE);
  const vertexCount = posSize / (3 * 4); // 3 floats * 4 bytes
  const cols = new Float32Array(vertexCount * 3);
  for (let i=0;i<vertexCount;i++){
    cols[i*3+0] = colorVec[0];
    cols[i*3+1] = colorVec[1];
    cols[i*3+2] = colorVec[2];
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, vboColor);
  gl.bufferData(gl.ARRAY_BUFFER, cols, gl.DYNAMIC_DRAW);
}

/* -----------------------------
   Camera & scene
   ----------------------------- */
let camEye = [0, 0, 8];
let camCenter = [0, 0, 0];
let camUp = [0,1,0];

function resizeCanvas(){
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || (window.innerWidth * 0.72);
  const cssH = canvas.clientHeight || window.innerHeight;
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  gl.viewport(0,0,canvas.width, canvas.height);
  // recalc projection cached below in draw
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

/* -----------------------------
   Draw scene (shadow pass + main pass)
   ----------------------------- */
const groundY = -1.8;
const lightDir = [0.5, 0.8, 0.3];

function drawScene(modelMat, drawShadow=true) {
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.03,0.05,0.07,1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // attributes
  gl.bindBuffer(gl.ARRAY_BUFFER, vboPos);
  gl.enableVertexAttribArray(loc.aPos);
  gl.vertexAttribPointer(loc.aPos, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, vboNormal);
  gl.enableVertexAttribArray(loc.aNormal);
  gl.vertexAttribPointer(loc.aNormal, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, vboColor);
  gl.enableVertexAttribArray(loc.aColor);
  gl.vertexAttribPointer(loc.aColor, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);

  // camera matrices
  const aspect = canvas.width / canvas.height;
  const proj = perspective(Math.PI/4, aspect, 0.1, 100.0);
  const view = lookAt(camEye, camCenter, camUp);
  gl.uniformMatrix4fv(loc.uProj, false, new Float32Array(proj));
  gl.uniformMatrix4fv(loc.uView, false, new Float32Array(view));

  // set common uniforms
  gl.uniform3fv(loc.uLightDir, new Float32Array(lightDir));
  gl.uniform3fv(loc.uViewPos, new Float32Array(camEye));
  gl.uniform1f(loc.uAmbient, 0.18);
  gl.uniform1f(loc.uSpecStrength, 0.8);
  gl.uniform1f(loc.uShininess, 28.0);


  // Main pass
  const normalM = inverseTransposeMat4(modelMat);
  gl.uniformMatrix4fv(loc.uModel, false, new Float32Array(modelMat));
  gl.uniformMatrix4fv(loc.uNormalMatrix, false, new Float32Array(normalM));
  gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_SHORT, 0);
}

/* -----------------------------
   Animation state machine
   ----------------------------- */
const sequenceDurations = {
  rot: 1000,    // ms per rotation step (will be scaled by 1/params.speed)
  scale: 900
};

let anim = {
  running: false,
  step: 0,
  t0: 0,
  yaw: 0,
  scale: 1,
  loopStart: 0
};

function easeInOut(t){ return t<0.5 ? 2*t*t : -1 + (4-2*t)*t; }

function startSequence() {
  if (anim.running) return;
  anim.running = true;
  anim.step = 0;
  anim.t0 = performance.now();
  anim.yaw = 0;
  anim.scale = 1;
  requestAnimationFrame(loop);
}
function stopSequence() { anim.running = false; }
function resetSequence() {
  anim.running = false;
  anim.step = 0;
  anim.t0 = 0;
  anim.yaw = 0;
  anim.scale = 1;
  // draw initial idle pose
  const model = multiply(scaleMat(1,1,1), rotateY(0));
  drawScene(model, true);
}

/* map steps:
   0: rotate 0 -> +pi
   1: +pi -> 0
   2: 0 -> -pi
   3: -pi -> 0
   4: scale up
   5: idle loop
*/
function loop(now) {
  resizeCanvas(); // ensure canvas size up to date
  if (!anim.running && params.mode === 'idle') {
    // idle gentle motion
    const t = performance.now() * 0.00025 * params.speed;
    const yaw = Math.sin(t) * 0.12;
    const tilt = Math.sin(t*1.3) * 0.05;
    const model = multiply(translation(0, Math.sin(t*1.2)*0.04, 0), multiply(scaleMat(1,1,1), multiply(rotateX(tilt), rotateY(yaw))));
    drawScene(model, true);
    requestAnimationFrame(loop);
    return;
  }

  if (!anim.running) {
    // not running and not idle mode: show default
    const model = multiply(scaleMat(1,1,1), rotateY(0));
    drawScene(model, true);
    requestAnimationFrame(loop);
    return;
  }

  const elapsed = now - anim.t0;
  const speedFactor = 1.0 / Math.max(0.0001, params.speed);
  const dur = sequenceDurations.rot * speedFactor;
  const sDur = sequenceDurations.scale * speedFactor;

  let model;

  if (anim.step <= 3) {
    const stepStart = anim.step * dur;
    const localT = Math.min(1, Math.max(0, (elapsed - stepStart) / dur));
    const eased = easeInOut(localT);
    let target = 0;
    if (anim.step === 0) target = Math.PI;
    if (anim.step === 1) target = 0;
    if (anim.step === 2) target = -Math.PI;
    if (anim.step === 3) target = 0;
    const from = (anim.step === 0) ? anim.yaw : anim.yaw;
    // For simplicity, interpolate from previous yaw (anim.yaw contains previous target)
    const prev = anim.yaw;
    const curYaw = prev + (target - prev) * eased;
    // on step end, set anim.yaw to target and advance
    if (localT >= 0.999) {
      anim.yaw = target;
      anim.step++;
      // adjust origin time to avoid accumulating elapsed
      anim.t0 = now - (elapsed - (stepStart + dur));
    } else {
      // temp yaw for this frame
      // do not overwrite anim.yaw until step completes
      model = rotateY(curYaw);
    }
    if (!model) model = rotateY(anim.yaw);
  } else if (anim.step === 4) {
    // scale up
    const stepStart = 4 * dur;
    const localT = Math.min(1, Math.max(0, (elapsed - stepStart) / sDur));
    const eased = easeInOut(localT);
    anim.scale = 1 + (2.6 - 1) * eased;
    model = multiply(scaleMat(anim.scale, anim.scale, anim.scale), rotateY(anim.yaw));
    if (localT >= 0.999) { anim.step = 5; anim.loopStart = now; anim.t0 = now; }
  } else {
    // idle loop: gentle rotation + bob
    const t = (now - anim.loopStart) * 0.001 * params.speed;
    const yaw = Math.sin(t * 0.6) * 0.6;
    const bob = Math.sin(t * 1.3) * 0.08;
    const rotY = rotateY(yaw);
    const rotX = rotateX(Math.sin(t*0.4)*0.06);
    const s = anim.scale || 2.6;
    let m = multiply(rotY, rotX);
    m = multiply(scaleMat(s,s,s), m);
    m = multiply(translation(0, bob, 0), m);
    model = m;
  }

  // if model hasn't been set (like during interpolation), build from anim.yaw
  if (!model) model = multiply(scaleMat(anim.scale||1,anim.scale||1,anim.scale||1), rotateY(anim.yaw || 0));

  drawScene(model, true);

  if (anim.running) requestAnimationFrame(loop);
}

/* -----------------------------
   UI wiring
   ----------------------------- */
elDepth.addEventListener('input', (e)=>{
  params.depth = parseFloat(e.target.value);
  depthVal.textContent = params.depth.toFixed(2);
  createGeometry(params.depth);
});

elSpeed.addEventListener('input', (e)=>{
  params.speed = parseFloat(e.target.value);
  speedVal.textContent = params.speed.toFixed(2);
});

elColor.addEventListener('input', (e)=>{
  params.color = hexToRgb(e.target.value);
  updateColorBuffer(params.color);
});

elMode.addEventListener('change', (e)=>{
  params.mode = e.target.value;
  if (params.mode === 'auto' && !anim.running) startSequence();
});

btnStart.addEventListener('click', ()=> {
  // single mode: if single trigger, start once then stop when hits idle
  startSequence();
});
btnStop.addEventListener('click', ()=> stopSequence());
btnReset.addEventListener('click', ()=> {
  resetSequence();
  // regenerate geometry to reset scale/origin
  createGeometry(params.depth);
});

window.addEventListener('keydown', (e)=>{
  if (e.code === 'Space') { if (anim.running) stopSequence(); else startSequence(); e.preventDefault(); }
  if (e.key === 'r' || e.key === 'R') { resetSequence(); }
});

/* -----------------------------
   Init: build geometry and start default mode
   ----------------------------- */
function hexToRgb(hex){
  if (hex[0] === '#') hex = hex.slice(1);
  const r = parseInt(hex.substring(0,2),16)/255;
  const g = parseInt(hex.substring(2,4),16)/255;
  const b = parseInt(hex.substring(4,6),16)/255;
  return [r,g,b];
}

createGeometry(params.depth);
resizeCanvas();
updateColorBuffer(params.color);

// Start automatically if mode is auto
if (params.mode === 'auto') setTimeout(()=> startSequence(), 500);

// draw initial frame
resetSequence();
