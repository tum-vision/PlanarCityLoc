/* Live LoD geometry, not paper-image animation. Camera motion is illustrative. */
(async () => {
  const figure = document.querySelector('.teaser-figure');
  if (!figure) return;
  const $ = selector => figure.querySelector(selector);
  const city = $('#teaser-city'), projection = $('#teaser-projection');
  const modelContext = city.getContext('2d'), queryContext = projection.getContext('2d');
  const play = $('#teaser-play'), slider = $('#teaser-offset'), status = $('#teaser-status');
  const steps = [...figure.querySelectorAll('[data-teaser-step]')];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let scene, query;
  try {
    if (!modelContext || !queryContext) throw Error('Canvas unavailable');
    const response = await fetch('assets/live-city/scene.json');
    if (!response.ok) throw Error('Model unavailable');
    scene = await response.json();
    query = new Image();
    query.src = 'assets/live-city/query.webp';
    await query.decode();
  } catch {
    status.textContent = 'The interactive model could not be loaded. Reload the page to retry.';
    figure.querySelectorAll('button,input').forEach(control => control.disabled = true);
    return;
  }
  const V = scene.vertices, F = scene.faces, P = scene.planes, R = scene.w2c, K = scene.K;
  const palette = ['#3478f6','#32ade6','#30b0c7','#46aa72','#e5a03b','#e27f60','#7092b0','#53a8a1'];
  const ns = 'http://www.w3.org/2000/svg';
  const svg = (tag, attributes, parent) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    parent.append(node);return node;
  };
  const curveRoot = $('#teaser-curves');
  const curveGroup = svg('g', {transform:'translate(52 28) scale(1.285206 1.427775) translate(-499.527 -40.504)'}, curveRoot);
  const curves = Object.fromEntries(Object.entries(window.PLANARCITYLOC_TEASER.curves).map(([name,d]) =>
    [name,svg('path',{d,class:'teaser-curve teaser-curve-'+name},curveGroup)]));
  const curveSamples = {};
  for (const name of Object.keys(curves)) {
    // The source artwork consists of polylines. Interpolate their native
    // vertices directly instead of repeatedly querying browser SVG geometry.
    const points = [...window.PLANARCITYLOC_TEASER.curves[name].matchAll(/[ML](-?[\d.]+) (-?[\d.]+)/g)].map(match => [Number(match[1]), Number(match[2])]);
    curveSamples[name] = Array.from({length:241}, (_, index) => {
      const x = 499.527 + (index/10 - 12 + 40)/80*314.352;
      let segment=0;
      while(segment<points.length-2 && points[segment+1][0]<x)segment++;
      const a=points[segment],b=points[segment+1];
      const y=a[1]+(b[1]-a[1])*(x-a[0])/(b[0]-a[0]);
      return [52+(x-499.527)*1.285206,28+(y-40.504)*1.427775];
    });
  }
  const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const subtract = (a,b) => a.map((v,i)=>v-b[i]);
  const clamp = (n,a=0,b=1) => Math.max(a,Math.min(b,n));
  const ease = n => n*n*(3-2*n);
  const cameraAxis = [R[0][0],R[0][1],R[0][2]];
  let yaw = Math.PI+.32, pitch=.67, zoom=1, selected=scene.suggested[0];
  let offset=8, time=0, mode=0, playing=!reduced.matches, visible=false, frame=null, previous=null;
  let modelHit=[], queryHit=[], dark=false;
  const themes = {
    light:{background:'#ffffff',grid:'#e8e8ed',edge:'#90949d',text:'#6e6e73',reference:'#ffffff',ground:'#f5f5f7'},
    dark:{background:'#1c1c1e',grid:'#36363a',edge:'#77808c',text:'#a1a1a6',reference:'#ffffff',ground:'#242428'}
  };
  const captions = [
    'Move the camera: similar building silhouettes can occur at different poses.',
    'Select a roof or façade. The same 3D plane is highlighted in the model and its image projection.',
    'Plane correspondences constrain the camera pose. Here, an illustrative motion brings it back to the reference alignment.'
  ];
  function selectPlane(index, manual=false) {
    selected=index;
    status.textContent=P[index].label+' selected · '+scene.frame;
    $('#teaser-roof').setAttribute('aria-pressed',String(index===scene.suggested[0]));
    $('#teaser-facade').setAttribute('aria-pressed',String(index===scene.suggested[1]));
    if(manual){mode=1;time=5000;playing=false;}
  }
  function canvasSize(canvas, ctx) {
    const width=canvas.clientWidth, height=canvas.clientHeight, ratio=Math.min(devicePixelRatio||1,2);
    if(canvas.width!==Math.round(width*ratio)||canvas.height!==Math.round(height*ratio)){
      canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
    }
    ctx.setTransform(ratio,0,0,ratio,0,0);
    return {width,height};
  }
  const polygon = (ctx, points) => {
    ctx.beginPath();ctx.moveTo(points[0][0],points[0][1]);
    for(let i=1;i<points.length;i++)ctx.lineTo(points[i][0],points[i][1]);
    ctx.closePath();
  };
  function shade(hex, factor) {
    const v=parseInt(hex.slice(1),16);
    return 'rgb('+[v>>16,(v>>8)&255,v&255].map(c=>Math.round(clamp(c*factor,0,255))).join(',')+')';
  }
  function renderModel(theme) {
    const ctx=modelContext,{width:w,height:h}=canvasSize(city,ctx);
    ctx.clearRect(0,0,w,h);ctx.fillStyle=theme.background;ctx.fillRect(0,0,w,h);
    const sy=Math.sin(yaw),cy=Math.cos(yaw),sp=Math.sin(pitch),cp=Math.cos(pitch);
    const scale=Math.min(w/215,h/166)*zoom;
    const view=[sy*cp,cy*cp,sp];
    const project=p=>{
      const x=p[0],y=p[1],z=p[2]-25;
      const rx=cy*x-sy*y, ry=sy*x+cy*y;
      return [w*.50+rx*scale,h*.56+(ry*sp-z*cp)*scale,ry*cp+z*sp];
    };
    ctx.strokeStyle=theme.grid;ctx.lineWidth=.7;
    for(let n=-75;n<=75;n+=15){
      for(const points of [[[-75,n,0],[75,n,0]],[[n,-75,0],[n,75,0]]]){
        const a=project(points[0]),b=project(points[1]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
      }
    }
    const points=V.map(project);
    const order=F.map((f,i)=>({i,z:(points[f[0]][2]+points[f[1]][2]+points[f[2]][2])/3}))
      .filter(({i})=>dot(P[F[i][3]].normal,view)>-.015).sort((a,b)=>a.z-b.z);
    modelHit=[];
    for(const {i} of order){
      const f=F[i],plane=P[f[3]],ps=f.slice(0,3).map(index=>points[index]);
      const light=.70+.30*Math.max(0,dot(plane.normal,[-.3,-.4,.86]));
      const base=f[3]===selected&&mode>0?theme.accent:mode>0?palette[f[3]%palette.length]:(dark?'#b4bac5':'#d3d7df');
      polygon(ctx,ps);ctx.fillStyle=shade(base,light);ctx.fill();
      ctx.strokeStyle=f[3]===selected&&mode>0?theme.accent:(dark?'#22262c':'#ffffff');
      ctx.lineWidth=f[3]===selected&&mode>0?.85:.28;ctx.stroke();
      modelHit.push({points:ps,id:f[3]});
    }
    // Actual dataset camera and a small frustum, translated by the illustrated offset.
    const camera=scene.camera.map((value,i)=>value+cameraAxis[i]*offset);
    const inverse=q=>[0,1,2].map(i=>camera[i]+R[0][i]*q[0]+R[1][i]*q[1]+R[2][i]*q[2]);
    const corners=[[-8,-6,13],[8,-6,13],[8,6,13],[-8,6,13]].map(inverse);
    const center=project(camera),corners2D=corners.map(project);
    ctx.strokeStyle=theme.accent;ctx.lineWidth=1.4;ctx.setLineDash([]);
    polygon(ctx,corners2D);ctx.stroke();
    for(const p of corners2D){ctx.beginPath();ctx.moveTo(center[0],center[1]);ctx.lineTo(p[0],p[1]);ctx.stroke();}
    ctx.fillStyle=theme.accent;ctx.beginPath();ctx.arc(center[0],center[1],4,0,Math.PI*2);ctx.fill();
    ctx.font='12px -apple-system, BlinkMacSystemFont, sans-serif';ctx.fillStyle=theme.text;
    ctx.fillText('Camera',center[0]+8,center[1]-6);
    const anchor=project(P[selected].center);
    if(mode>0){ctx.fillStyle=theme.accent;ctx.strokeStyle=theme.background;ctx.lineWidth=2;ctx.beginPath();ctx.arc(anchor[0],anchor[1],5,0,Math.PI*2);ctx.fill();ctx.stroke();}
    return anchor;
  }
  function renderQuery(theme) {
    const ctx=queryContext,{width:w,height:h}=canvasSize(projection,ctx);
    ctx.clearRect(0,0,w,h);ctx.fillStyle=theme.background;ctx.fillRect(0,0,w,h);
    const scale=Math.min((w-20)/scene.width,(h-20)/scene.height),iw=scene.width*scale,ih=scene.height*scale;
    const ox=(w-iw)/2,oy=(h-ih)/2;
    ctx.drawImage(query,ox,oy,iw,ih);
    const project=(p,translation=offset)=>{
      const q=[0,1,2].map(i=>dot(R[i],p)+R[i][3]);q[0]-=translation;
      return [ox+(K[0][0]*q[0]/q[2]+K[0][2])*scale,oy+(K[1][1]*q[1]/q[2]+K[1][2])*scale,q[2]];
    };
    const points=V.map(p=>project(p));
    const camera=scene.camera.map((v,i)=>v+cameraAxis[i]*offset);
    const order=F.map((f,i)=>({i,z:(points[f[0]][2]+points[f[1]][2]+points[f[2]][2])/3}))
      .filter(({i,z})=>z>1&&dot(P[F[i][3]].normal,subtract(camera,P[F[i][3]].center))>0)
      .sort((a,b)=>b.z-a.z);
    queryHit=[];
    ctx.save();ctx.beginPath();ctx.rect(ox,oy,iw,ih);ctx.clip();
    for(const {i} of order){
      const f=F[i],ps=f.slice(0,3).map(index=>points[index]),chosen=f[3]===selected;
      polygon(ctx,ps);
      ctx.fillStyle=mode===0?theme.accent+'47':chosen?theme.accent+'7a':shade(palette[f[3]%palette.length],1);
      ctx.globalAlpha=mode===0||chosen?1:.15;ctx.fill();ctx.globalAlpha=1;
      ctx.strokeStyle=chosen&&mode>0?theme.accent:mode===0?'rgba(255,255,255,.15)':'rgba(255,255,255,.32)';
      ctx.lineWidth=chosen&&mode>0?1.7:.3;ctx.stroke();
      queryHit.push({points:ps,id:f[3]});
    }
    if(mode>0){
      ctx.setLineDash([4,3]);ctx.strokeStyle=theme.reference;ctx.lineWidth=1.3;
      F.filter(f=>f[3]===selected).forEach(f=>{polygon(ctx,f.slice(0,3).map(i=>project(V[i],0)));ctx.stroke();});
      ctx.setLineDash([]);
    }
    const anchor=project(P[selected].center);
    if(mode>0){ctx.fillStyle=theme.accent;ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(anchor[0],anchor[1],5,0,Math.PI*2);ctx.fill();ctx.stroke();}
    ctx.restore();
    return anchor[0]>=ox&&anchor[0]<=ox+iw&&anchor[1]>=oy&&anchor[1]<=oy+ih?anchor:null;
  }
  function renderConnection(a,b) {
    const overlay=$('#teaser-link-overlay');
    if(mode===0||!b){overlay.style.opacity='0';return;}
    overlay.style.opacity='1';
    const bounds=$('#teaser-viewports').getBoundingClientRect(),left=city.getBoundingClientRect(),right=projection.getBoundingClientRect();
    overlay.setAttribute('viewBox','0 0 '+bounds.width+' '+bounds.height);
    const x1=left.left-bounds.left+a[0],y1=left.top-bounds.top+a[1],x2=right.left-bounds.left+b[0],y2=right.top-bounds.top+b[1],mid=(x1+x2)/2;
    const path=overlay.querySelector('path');
    path.setAttribute('d','M'+x1+' '+y1+'C'+mid+' '+y1+' '+mid+' '+y2+' '+x2+' '+y2);
    path.style.strokeDashoffset=String(-time/65);
    const t=(time%2200)/2200,u=1-t;
    const packet=overlay.querySelector('circle');
    packet.setAttribute('cx',u*u*u*x1+3*u*u*t*mid+3*u*t*t*mid+t*t*t*x2);
    packet.setAttribute('cy',u*u*u*y1+3*u*u*t*y1+3*u*t*t*y2+t*t*t*y2);
    packet.style.opacity=playing?'1':'0';
  }
  function render() {
    dark=document.documentElement.dataset.theme==='dark';
    const theme=themes[dark?'dark':'light'];
    theme.accent=getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
    figure.dataset.stage=String(mode);figure.dataset.playing=String(playing);
    figure.dataset.offset=offset.toFixed(2);figure.dataset.selectedPlane=String(selected);
    play.querySelector('span').textContent=playing?'Pause':'Play';
    play.setAttribute('aria-label',(playing?'Pause':'Play')+' teaser animation');
    steps.forEach((button,index)=>button.setAttribute('aria-pressed',String(mode===index)));
    $('#teaser-explanation').textContent=captions[mode];
    slider.value=String(offset);
    slider.setAttribute('aria-valuetext',offset.toFixed(1)+' metres from the reference camera');
    $('#teaser-offset-value').textContent=(offset>=0?'+':'')+offset.toFixed(1)+' m';
    renderConnection(renderModel(theme),renderQuery(theme));
    const sample=curveSamples[mode===0?'v2':'ours'][Math.round(clamp(offset,-12,12)*10+120)];
    const marker=$('#teaser-pose'),line=$('#teaser-refinement');
    marker.setAttribute('cx',sample[0]);marker.setAttribute('cy',sample[1]);
    line.setAttribute('x1',sample[0]);line.setAttribute('x2',sample[0]);line.setAttribute('y1',28);line.setAttribute('y2',284);
    for(const[name,curve]of Object.entries(curves))curve.style.opacity=mode===0?(name==='ours'?'.35':'1'):(name==='ours'?'1':'.45');
  }
  function animate() {
    mode=Math.min(2,Math.floor(time/5000));
    const t=(time%5000)/5000;
    if(mode===0)offset=8*Math.cos(t*Math.PI*2);
    if(mode===1){
      offset=6;
      const next=scene.suggested[Math.floor(t*4)%2];
      if(next!==selected)selectPlane(next);
    }
    if(mode===2)offset=6*(1-ease(clamp(t/.8)));
    yaw=Math.PI+.32+Math.sin(time/15000*Math.PI*2)*.3;
  }
  function tick(now) {
    frame=null;
    if(!playing||!visible||document.hidden){previous=null;return;}
    if(previous===null)previous=now;
    const delta=now-previous;
    if(delta>=32){time=(time+Math.min(delta,100))%15000;previous=now;animate();render();}
    frame=requestAnimationFrame(tick);
  }
  function schedule() {
    if(frame!==null)cancelAnimationFrame(frame);
    frame=null;previous=null;render();
    if(playing&&visible&&!document.hidden)frame=requestAnimationFrame(tick);
  }
  function manual(){playing=false;schedule();}
  play.addEventListener('click',()=>{playing=!playing;schedule();});
  $('#teaser-replay').addEventListener('click',()=>{time=0;mode=0;offset=8;playing=true;selectPlane(scene.suggested[0]);schedule();});
  steps.forEach((button,index)=>button.addEventListener('click',()=>{
    time=index*5000;mode=index;offset=index===0?8:6;playing=!reduced.matches;
    if(index===1)selectPlane(scene.suggested[0]);schedule();
  }));
  slider.addEventListener('input',()=>{offset=Number(slider.value);manual();});
  $('#teaser-roof').addEventListener('click',()=>{selectPlane(scene.suggested[0],true);schedule();});
  $('#teaser-facade').addEventListener('click',()=>{
    selectPlane(scene.suggested[1],true);const n=P[selected].normal;yaw=Math.atan2(n[0],n[1]);schedule();
  });
  $('#teaser-reset').addEventListener('click',()=>{yaw=Math.PI+.32;pitch=.67;zoom=1;manual();});
  function inside(p,triangle) {
    const[a,b,c]=triangle;
    const sign=(a,b)=>(p[0]-b[0])*(a[1]-b[1])-(a[0]-b[0])*(p[1]-b[1]);
    const s=[sign(a,b),sign(b,c),sign(c,a)];return !(s.some(v=>v<0)&&s.some(v=>v>0));
  }
  function pick(event,canvas,hits) {
    const bounds=canvas.getBoundingClientRect(),p=[event.clientX-bounds.left,event.clientY-bounds.top];
    const hit=[...hits].reverse().find(hit=>inside(p,hit.points));
    if(hit){selectPlane(hit.id,true);schedule();}
  }
  let drag=null;
  city.addEventListener('pointerdown',event=>{
    if(event.button!==0)return;
    drag={x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,moved:false};city.setPointerCapture(event.pointerId);
  });
  city.addEventListener('pointermove',event=>{
    if(!drag)return;
    const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
    if(Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)>4)drag.moved=true;
    if(drag.moved){yaw+=dx*.008;pitch=clamp(pitch+dy*.006,.18,1.3);manual();}
    drag.x=event.clientX;drag.y=event.clientY;
  });
  city.addEventListener('pointerup',event=>{if(drag&&!drag.moved)pick(event,city,modelHit);drag=null;});
  city.addEventListener('pointercancel',()=>drag=null);
  projection.addEventListener('click',event=>pick(event,projection,queryHit));
  const selectable=P.map((p,i)=>({p,i})).filter(({p})=>p.area>15).map(({i})=>i);
  function keyboard(event) {
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','=','Enter','Home'].includes(event.key))return;
    event.preventDefault();
    if(event.key==='ArrowLeft')yaw-=.12;if(event.key==='ArrowRight')yaw+=.12;
    if(event.key==='ArrowUp')pitch=clamp(pitch+.08,.18,1.3);if(event.key==='ArrowDown')pitch=clamp(pitch-.08,.18,1.3);
    if(event.key==='+'||event.key==='=')zoom=clamp(zoom*1.12,.6,2);
    if(event.key==='-')zoom=clamp(zoom/1.12,.6,2);
    if(event.key==='Home'){yaw=Math.PI+.32;pitch=.67;zoom=1;}
    if(event.key==='Enter')selectPlane(selectable[(selectable.indexOf(selected)+1)%selectable.length],true);
    manual();
  }
  city.addEventListener('keydown',keyboard);projection.addEventListener('keydown',keyboard);
  reduced.addEventListener('change',()=>{if(reduced.matches)manual();});
  document.addEventListener('visibilitychange',schedule);
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;schedule();},{threshold:.05}).observe(figure);
  new ResizeObserver(schedule).observe($('#teaser-viewports'));
  new MutationObserver(schedule).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  selectPlane(selected);schedule();
})();
