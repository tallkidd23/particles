/* ============================================================
   AIR CURRENT SIMULATOR
   Grid-based "Stable Fluids" solver (Jos Stam method) driving
   a particle overlay for wind visualization. Walls/corners are
   solid grid cells; vortices emerge naturally from the pressure
   projection step when flow can't escape a boundary.
   ============================================================ */

(function () {
  const canvas = document.getElementById('simCanvas');
  const ctx = canvas.getContext('2d');
  const SIZE = canvas.width; // square canvas
  const N = 72;              // interior grid resolution
  const CELL = SIZE / N;

  function IX(i, j) { return i + (N + 2) * j; }

  // ---------- Fluid state ----------
  let dt = 0.15;
  let diff = 0.0000005;
  let visc = 0.00005;

  const total = (N + 2) * (N + 2);
  let u = new Float32Array(total);
  let v = new Float32Array(total);
  let u0 = new Float32Array(total);
  let v0 = new Float32Array(total);
  let curl = new Float32Array(total);
  let solid = new Uint8Array(total);

  const VORTICITY_EPS = 0.09;

  function clearFluid() {
    u.fill(0); v.fill(0); u0.fill(0); v0.fill(0); curl.fill(0);
  }

  function clearWalls() {
    solid.fill(0);
  }

  function setSolidRect(i0, j0, i1, j1, value) {
    for (let j = Math.max(1, j0); j <= Math.min(N, j1); j++) {
      for (let i = Math.max(1, i0); i <= Math.min(N, i1); i++) {
        solid[IX(i, j)] = value;
      }
    }
  }

  // ---------- Boundary handling ----------
  function setBnd(b, x) {
    for (let i = 1; i <= N; i++) {
      x[IX(0, i)]     = b === 1 ? -x[IX(1, i)] : x[IX(1, i)];
      x[IX(N + 1, i)]  = b === 1 ? -x[IX(N, i)] : x[IX(N, i)];
      x[IX(i, 0)]      = b === 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
      x[IX(i, N + 1)]  = b === 2 ? -x[IX(i, N)] : x[IX(i, N)];
    }
    x[IX(0, 0)]         = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, N + 1)]     = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
    x[IX(N + 1, 0)]     = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
    x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);

    // Solid interior cells hold no velocity/pressure of their own.
    for (let k = 0; k < total; k++) {
      if (solid[k]) x[k] = 0;
    }

    // Cancel velocity components that would push fluid into a wall.
    if (b === 1 || b === 2) {
      for (let j = 1; j <= N; j++) {
        for (let i = 1; i <= N; i++) {
          const idx = IX(i, j);
          if (solid[idx]) continue;
          if (b === 1) {
            if (solid[IX(i + 1, j)] && x[idx] > 0) x[idx] = 0;
            if (solid[IX(i - 1, j)] && x[idx] < 0) x[idx] = 0;
          } else {
            if (solid[IX(i, j + 1)] && x[idx] > 0) x[idx] = 0;
            if (solid[IX(i, j - 1)] && x[idx] < 0) x[idx] = 0;
          }
        }
      }
    }
  }

  function linSolve(b, x, x0, a, c) {
    const cRecip = 1 / c;
    for (let k = 0; k < 18; k++) {
      for (let j = 1; j <= N; j++) {
        for (let i = 1; i <= N; i++) {
          const idx = IX(i, j);
          if (solid[idx]) { x[idx] = 0; continue; }
          x[idx] = (x0[idx] + a * (
            x[IX(i + 1, j)] + x[IX(i - 1, j)] +
            x[IX(i, j + 1)] + x[IX(i, j - 1)]
          )) * cRecip;
        }
      }
      setBnd(b, x);
    }
  }

  function diffuse(b, x, x0, diffRate) {
    const a = dt * diffRate * N * N;
    linSolve(b, x, x0, a, 1 + 4 * a);
  }

  function advect(b, d, d0, velU, velV) {
    const dt0 = dt * N;
    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        const idx = IX(i, j);
        if (solid[idx]) { d[idx] = 0; continue; }
        let x = i - dt0 * velU[idx];
        let y = j - dt0 * velV[idx];
        if (x < 0.5) x = 0.5;
        if (x > N + 0.5) x = N + 0.5;
        if (y < 0.5) y = 0.5;
        if (y > N + 0.5) y = N + 0.5;
        const i0 = Math.floor(x), i1 = i0 + 1;
        const j0 = Math.floor(y), j1 = j0 + 1;
        const s1 = x - i0, s0 = 1 - s1;
        const t1 = y - j0, t0 = 1 - t1;
        d[idx] =
          s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
          s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
      }
    }
    setBnd(b, d);
  }

  function project(velU, velV, p, div) {
    const h = 1 / N;
    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        const idx = IX(i, j);
        div[idx] = solid[idx] ? 0 : -0.5 * h * (
          velU[IX(i + 1, j)] - velU[IX(i - 1, j)] +
          velV[IX(i, j + 1)] - velV[IX(i, j - 1)]
        );
        p[idx] = 0;
      }
    }
    setBnd(0, div);
    setBnd(0, p);
    linSolve(0, p, div, 1, 4);

    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        const idx = IX(i, j);
        if (solid[idx]) continue;
        velU[idx] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
        velV[idx] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
      }
    }
    setBnd(1, velU);
    setBnd(2, velV);
  }

  // Vorticity confinement: keeps rotational eddies alive instead of
  // letting numerical diffusion smooth them away.
  function vorticityConfinement() {
    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        const idx = IX(i, j);
        curl[idx] = solid[idx] ? 0 :
          (v[IX(i + 1, j)] - v[IX(i - 1, j)]) * 0.5 -
          (u[IX(i, j + 1)] - u[IX(i, j - 1)]) * 0.5;
      }
    }
    for (let j = 2; j < N; j++) {
      for (let i = 2; i < N; i++) {
        const idx = IX(i, j);
        if (solid[idx]) continue;
        let dx = (Math.abs(curl[IX(i + 1, j)]) - Math.abs(curl[IX(i - 1, j)])) * 0.5;
        let dy = (Math.abs(curl[IX(i, j + 1)]) - Math.abs(curl[IX(i, j - 1)])) * 0.5;
        const len = Math.sqrt(dx * dx + dy * dy) + 1e-5;
        dx /= len; dy /= len;
        const w = curl[idx];
        u[idx] += VORTICITY_EPS * (dy * -w) * dt;
        v[idx] += VORTICITY_EPS * (dx * w) * dt;
      }
    }
  }

  function velStep() {
    diffuse(1, u0, u, visc);
    diffuse(2, v0, v, visc);
    [u, u0] = [u0, u];
    [v, v0] = [v0, v];

    vorticityConfinement();
    project(u, v, u0, v0);

    advect(1, u0, u, u, v);
    advect(2, v0, v, u, v);
    [u, u0] = [u0, u];
    [v, v0] = [v0, v];

    project(u, v, u0, v0);
  }

  function addForce(gx, gy, fx, fy, radius) {
    const r = radius;
    for (let j = Math.max(1, gy - r); j <= Math.min(N, gy + r); j++) {
      for (let i = Math.max(1, gx - r); i <= Math.min(N, gx + r); i++) {
        const d = Math.hypot(i - gx, j - gy);
        if (d > r) continue;
        const idx = IX(i, j);
        if (solid[idx]) continue;
        const falloff = 1 - d / r;
        u[idx] += fx * falloff;
        v[idx] += fy * falloff;
      }
    }
  }

  function sampleVelocity(px, py) {
    let x = px / CELL, y = py / CELL;
    if (x < 0.5) x = 0.5; if (x > N + 0.5) x = N + 0.5;
    if (y < 0.5) y = 0.5; if (y > N + 0.5) y = N + 0.5;
    const i0 = Math.floor(x), i1 = i0 + 1;
    const j0 = Math.floor(y), j1 = j0 + 1;
    const s1 = x - i0, s0 = 1 - s1;
    const t1 = y - j0, t0 = 1 - t1;
    const su =
      s0 * (t0 * u[IX(i0, j0)] + t1 * u[IX(i0, j1)]) +
      s1 * (t0 * u[IX(i1, j0)] + t1 * u[IX(i1, j1)]);
    const sv =
      s0 * (t0 * v[IX(i0, j0)] + t1 * v[IX(i0, j1)]) +
      s1 * (t0 * v[IX(i1, j0)] + t1 * v[IX(i1, j1)]);
    return [su, sv];
  }

  // ---------- Particles ----------
  let particles = [];
  function makeParticle() {
    return {
      x: Math.random() * SIZE,
      y: Math.random() * SIZE,
      life: 60 + Math.random() * 120
    };
  }
  function setParticleCount(n) {
    if (n > particles.length) {
      while (particles.length < n) particles.push(makeParticle());
    } else {
      particles.length = n;
    }
  }

  function isSolidAtPixel(px, py) {
    const i = Math.floor(px / CELL) + 1;
    const j = Math.floor(py / CELL) + 1;
    if (i < 1 || i > N || j < 1 || j > N) return false;
    return !!solid[IX(i, j)];
  }

  function stepParticles() {
    for (const p of particles) {
      const [vx, vy] = sampleVelocity(p.x, p.y);
      p.x += vx * N * 1.6;
      p.y += vy * N * 1.6;
      p.life -= 1;
      if (
        p.life <= 0 || p.x < 0 || p.x > SIZE || p.y < 0 || p.y > SIZE ||
        isSolidAtPixel(p.x, p.y)
      ) {
        Object.assign(p, makeParticle());
      }
    }
  }

  // ---------- Rendering ----------
  function drawWalls() {
    ctx.fillStyle = '#274049';
    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        if (solid[IX(i, j)]) {
          ctx.fillRect((i - 1) * CELL, (j - 1) * CELL, CELL + 1, CELL + 1);
        }
      }
    }
  }

  function drawParticles() {
    for (const p of particles) {
      const [vx, vy] = sampleVelocity(p.x, p.y);
      const speed = Math.hypot(vx, vy) * N;
      const hue = Math.max(0, 200 - Math.min(200, speed * 260));
      ctx.fillStyle = `hsl(${hue}, 100%, ${60 + Math.min(30, speed * 40)}%)`;
      ctx.fillRect(p.x, p.y, 1.6, 1.6);
    }
  }

  let fadeAlpha = 0.08;
  function render() {
    ctx.fillStyle = `rgba(0,0,0,${fadeAlpha})`;
    ctx.fillRect(0, 0, SIZE, SIZE);
    drawParticles();
    drawWalls();
  }

  function loop() {
    velStep();
    stepParticles();
    render();
    requestAnimationFrame(loop);
  }

  // ---------- Interaction ----------
  let mode = 'blow'; // 'blow' | 'wall' | 'erase'
  let dragging = false;
  let lastX = 0, lastY = 0;

  function canvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = SIZE / rect.width;
    const scaleY = SIZE / rect.height;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return [(clientX - rect.left) * scaleX, (clientY - rect.top) * scaleY];
  }

  let blowerStrength = 18;

  function handleDown(e) {
    e.preventDefault();
    dragging = true;
    [lastX, lastY] = canvasPos(e);
  }
  function handleMove(e) {
    if (!dragging) return;
    e.preventDefault();
    const [x, y] = canvasPos(e);
    const gx = Math.floor(x / CELL) + 1;
    const gy = Math.floor(y / CELL) + 1;

    if (mode === 'blow') {
      const dx = (x - lastX) * 0.9;
      const dy = (y - lastY) * 0.9;
      addForce(gx, gy, dx * blowerStrength * 0.02, dy * blowerStrength * 0.02, 5);
    } else if (mode === 'wall') {
      setSolidRect(gx - 1, gy - 1, gx + 1, gy + 1, 1);
    } else if (mode === 'erase') {
      setSolidRect(gx - 1, gy - 1, gx + 1, gy + 1, 0);
    }
    lastX = x; lastY = y;
  }
  function handleUp() { dragging = false; }

  canvas.addEventListener('mousedown', handleDown);
  canvas.addEventListener('mousemove', handleMove);
  window.addEventListener('mouseup', handleUp);
  canvas.addEventListener('touchstart', handleDown, { passive: false });
  canvas.addEventListener('touchmove', handleMove, { passive: false });
  canvas.addEventListener('touchend', handleUp);

  // ---------- Controls ----------
  const modeButtons = {
    blow: document.getElementById('modeBlow'),
    wall: document.getElementById('modeWall'),
    erase: document.getElementById('modeErase')
  };
  const hintText = document.getElementById('hintText');
  const hints = {
    blow: 'Drag anywhere on the canvas to blast air in that direction.',
    wall: 'Drag to paint solid wall cells. Air will deflect around them.',
    erase: 'Drag over walls to remove them.'
  };
  function setMode(m) {
    mode = m;
    Object.entries(modeButtons).forEach(([k, btn]) => btn.classList.toggle('active', k === m));
    hintText.textContent = hints[m];
  }
  modeButtons.blow.onclick = () => setMode('blow');
  modeButtons.wall.onclick = () => setMode('wall');
  modeButtons.erase.onclick = () => setMode('erase');

  document.getElementById('presetClear').onclick = () => { clearWalls(); };
  document.getElementById('presetWall').onclick = () => {
    clearWalls();
    setSolidRect(Math.floor(N * 0.55), Math.floor(N * 0.15), Math.floor(N * 0.58), Math.floor(N * 0.85), 1);
  };
  document.getElementById('presetCorner').onclick = () => {
    clearWalls();
    // L-shaped corner in the bottom-right: a vertical + horizontal wall meeting.
    setSolidRect(Math.floor(N * 0.62), Math.floor(N * 0.35), Math.floor(N * 0.65), N, 1);
    setSolidRect(Math.floor(N * 0.62), Math.floor(N * 0.62), N, Math.floor(N * 0.65), 1);
  };

  const blowerSlider = document.getElementById('blowerStrength');
  const blowerVal = document.getElementById('blowerVal');
  blowerSlider.oninput = () => {
    blowerStrength = +blowerSlider.value;
    blowerVal.textContent = blowerStrength;
  };

  const viscSlider = document.getElementById('viscosity');
  const viscVal = document.getElementById('viscVal');
  viscSlider.oninput = () => {
    visc = (+viscSlider.value) * 0.00001;
    viscVal.textContent = visc.toFixed(6);
  };

  const particleSlider = document.getElementById('particleCount');
  const particleVal = document.getElementById('particleVal');
  particleSlider.oninput = () => {
    setParticleCount(+particleSlider.value);
    particleVal.textContent = particleSlider.value;
  };

  const fadeSlider = document.getElementById('trailFade');
  const fadeVal = document.getElementById('fadeVal');
  fadeSlider.oninput = () => {
    fadeAlpha = (+fadeSlider.value) / 100;
    fadeVal.textContent = fadeAlpha.toFixed(2);
  };

  document.getElementById('resetBtn').onclick = () => {
    clearFluid();
    setParticleCount(0);
    setParticleCount(+particleSlider.value);
  };

  // ---------- Init ----------
  setParticleCount(+particleSlider.value);
  document.getElementById('presetCorner').click();
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, SIZE, SIZE);
  loop();
})();
