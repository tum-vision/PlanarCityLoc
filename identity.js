/* Animate the existing logo layers; render real LoD geometry behind the title. */
(() => {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const marks = [...document.querySelectorAll('.animated-mark')];
  // Roof vertices traced in the source logo's 512 × 512 coordinate system.
  // Share them between the highlight and rays so neither spills onto a façade.
  const roof = [[117,254],[217,275],[290,322],[200,345]];
  const rotors = [[313,85],[289,136],[451,114],[425,167]];
  const rotorCutouts = ['M255 91 357 62 371 67 365 78 263 106 252 101Z', 'M243 116 259 115 329 145 330 153 316 155 245 128Z', 'M414 89 429 89 500 124 499 137 481 137 414 104Z', 'M378 174 468 149 478 154 473 165 393 188 377 184Z'];
  marks.forEach((mark, index) => {
    const id = `identity-${index}`;
    mark.innerHTML = `<svg viewBox="45 28 490 475" aria-hidden="true"><defs>
      <clipPath id="${id}-building"><path d="M70 295 110 247 219 273 299 315 300 422 185 480 68 423Z"/></clipPath>
      <clipPath id="${id}-drone"><path d="M245 84 378 49 416 86 506 103 515 161 478 189 422 197 374 177 353 166 340 158 333 145 311 145 292 162 269 152 268 143 237 137 231 118Z"/></clipPath>
      <clipPath id="${id}-rotor"><path d="${rotorCutouts[0]}"/></clipPath>
      <mask id="${id}-body" maskUnits="userSpaceOnUse" x="0" y="0" width="512" height="512"><rect width="512" height="512" fill="white"/>${rotorCutouts.map(d=>`<path d="${d}" fill="black"/>`).join('')}</mask></defs>
      <image href="assets/logo-512.png" width="512" height="512" clip-path="url(#${id}-building)"/>
      <path class="logo-plane" d="M${roof.map(p=>p.join(' ')).join('L')}Z"/>
      <g class="logo-rays">${roof.map(()=>'<path class="logo-ray"/>').join('')}</g>
      <g class="logo-drone"><image href="assets/logo-512.png" width="512" height="512" clip-path="url(#${id}-drone)" mask="url(#${id}-body)"/>
        ${rotors.map(([x,y])=>`<g class="logo-rotor" transform="translate(${x} ${y}) matrix(1 -.28 0 .48 0 0)"><circle class="logo-rotor-disc" r="51"/><g class="logo-blades"><g transform="matrix(1 .583333 0 2.083333 0 0) translate(-313 -85)"><image href="assets/logo-512.png" width="512" height="512" clip-path="url(#${id}-rotor)"/></g></g><circle class="logo-rotor-hub" r="6"/></g>`).join('')}
      </g>
      <g class="logo-points">${roof.map(([x,y])=>`<circle class="logo-point" cx="${x}" cy="${y}" r="5"/>`).join('')}</g></svg>`;
  });
  const canvas = document.querySelector('#hero-geometry'), ctx = canvas?.getContext('2d');
  // Logo instances always autoplay while visible, without a playback control.
  // Reduced motion still applies to the background city, independently.
  let scene, raf = null, logoTime = 0, geometryTime = 0, previous = 0;
  const visible = new Set();
  const targets = [...marks, ...(canvas ? [canvas] : [])];
  function drawLogo(mark) {
    const dx = Math.sin(logoTime / 1200) * 22, dy = Math.sin(logoTime / 700) * 30;
    mark.querySelector('.logo-drone').setAttribute('transform', `translate(${dx} ${dy})`);
    mark.querySelectorAll('.logo-blades').forEach((blades,i)=>blades.setAttribute('transform', `rotate(${(logoTime * .48 * (i%2 ? -1 : 1) + i*47)%360})`));
    roof.forEach(([x,y],i) => mark.querySelectorAll('.logo-ray')[i].setAttribute('d', `M${354+dx} ${155+dy}L${x} ${y}`));
  }
  function drawGeometry() {
    if (!ctx || !scene) return;
    const w = canvas.clientWidth, h = canvas.clientHeight, dpr = Math.min(devicePixelRatio || 1, 1.5);
    if (canvas.width !== Math.round(w*dpr) || canvas.height !== Math.round(h*dpr)) { canvas.width = Math.round(w*dpr); canvas.height = Math.round(h*dpr); }
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    const yaw = 3.45 + Math.sin(geometryTime / 15000) * .12, sy = Math.sin(yaw), cy = Math.cos(yaw), scale = Math.min(w/150,h/135);
    const vertices = scene.vertices.map(([x,y,z]) => {const ry = sy*x+cy*y;return [w*.5+(cy*x-sy*y)*scale,h*.55+(ry*.57-(z-24)*.82)*scale,ry*.82+z*.57];});
    const colors = document.documentElement.dataset.theme === 'dark' ? ['#a56962','#754844','#8e6963','#938b85'] : ['#e5b6aa','#bf8278','#cea598','#c4b8aa'];
    const faces = scene.faces.map(f=>({f,z:(vertices[f[0]][2]+vertices[f[1]][2]+vertices[f[2]][2])/3})).sort((a,b)=>a.z-b.z);
    for (const {f} of faces) {
      const n = scene.planes[f[3]].normal;
      if (n[0]*sy*.82+n[1]*cy*.82+n[2]*.57 < 0) continue;
      ctx.beginPath();f.slice(0,3).forEach((i,j)=>j?ctx.lineTo(vertices[i][0],vertices[i][1]):ctx.moveTo(vertices[i][0],vertices[i][1]));ctx.closePath();
      ctx.fillStyle = colors[f[3]%colors.length];ctx.fill();ctx.strokeStyle = '#ffffff68';ctx.lineWidth=.4;ctx.stroke();
    }
  }
  function draw() { marks.forEach(mark => {if (visible.has(mark)) drawLogo(mark);});if (visible.has(canvas)) drawGeometry(); }
  function shouldAnimate() { return !document.hidden && (marks.some(mark=>visible.has(mark)) || (!reduced.matches && visible.has(canvas))); }
  function tick(now) {
    raf = null;
    if (!shouldAnimate()) return;
    if (now-previous >= 32) {
      const elapsed = Math.min(now-previous,100); previous=now;
      logoTime+=elapsed; marks.forEach(mark=>{if(visible.has(mark))drawLogo(mark);});
      if (!reduced.matches && visible.has(canvas)) { geometryTime+=elapsed; drawGeometry(); }
    }
    raf=requestAnimationFrame(tick);
  }
  function schedule() { if(raf!==null)cancelAnimationFrame(raf);raf=null;draw();previous=performance.now();if(shouldAnimate())raf=requestAnimationFrame(tick); }
  marks.forEach(drawLogo);
  const observer = new IntersectionObserver(entries=>{entries.forEach(entry=>entry.isIntersecting?visible.add(entry.target):visible.delete(entry.target));schedule();});
  targets.forEach(target=>observer.observe(target));
  document.addEventListener('visibilitychange', schedule);reduced.addEventListener('change',schedule);
  new MutationObserver(schedule).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  if(canvas)new ResizeObserver(schedule).observe(canvas);
  fetch('assets/live-city/scene.json').then(response=>{if(!response.ok)throw Error('Unavailable');return response.json();}).then(value=>{scene=value;schedule();}).catch(()=>{});
})();
