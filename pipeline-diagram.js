/* Native reconstruction of Fig. 2: geometry and text are SVG, while thumbnails
   are complete original image objects extracted from the authors' pipeline PDF.
   Animation values are illustrative, not measurements or recorded inference. */
(() => {
  const root = document.querySelector('#pipeline-explorer');
  if (!root) return;
  const host = root.querySelector('#pipeline-canvas');
  const asset = (name) => `assets/pipeline/${name}.webp`;
  const stages = [
    { tag: '(a) Query geometry', title: 'Geometry and semantics extraction',
      text: 'The camera-ready configuration uses MoGe-2 ViT-S for depth and surface normals. SAM 3 supplies a building mask, and a Sobel filter on the normals extracts structural edges inside the building region.',
      input: 'RGB query Iq', output: 'Depth Dq · normals Nq · mask Mq · edges Eq', box: [16,202,260,594] },
    { tag: '(b) Plane extraction', title: 'From pixels to planar regions',
      text: 'Four-connected region growing joins neighboring building pixels only when their normals agree and their relative depths are consistent. The same rule is applied to the rendered model. Each region retains a centroid and a normal.',
      input: 'Building masks, normals, and depth', output: 'Image patches Pj and rendered LoD patches Qi', box: [294,202,240,594] },
    { tag: '(c) Coarse initialization', title: 'Constrain the search before matching planes',
      text: 'A Z-prescreen first constrains altitude at the GPS location. A 4D cost volume searches translation and yaw; multi-scale particle filtering refines the candidate using mask, edge, and depth agreement.',
      input: 'GPS prior ξp · query cues · rendered cues', output: 'Coarse pose ξf', box: [558,202,866,228] },
    { tag: '(d) 3D planar grounding', title: 'Re-render, associate planes, and solve PnP',
      text: 'Overlap voting tolerates fragmented query planes; centroid consistency then gives finer associations. Rendered depth lifts model-plane centroids to world coordinates. PnP with LO-RANSAC refines the pose, then the model is rendered again to update the matches.',
      input: '2D centroids p̄j · 3D centroids Xi · intrinsics K', output: 'Refined candidates ξ(1), …, ξ(N)', box: [558,458,662,338] },
    { tag: '(e) Final selection', title: 'Select the pose using complementary evidence',
      text: 'The coarse and refined poses are ranked by eight criteria: composite score, mask recall, mask IoU, edge alignment, depth correlation, normal error, reprojection error, and plane-centroid distance. Rank aggregation selects the final estimate.',
      input: 'Coarse pose and all PnP candidates', output: 'Final 6-DoF camera pose ξ*', box: [1244,458,180,338] }
  ];
  const txt = (x,y,s,kind='p-label',extra='') => `<text x="${x}" y="${y}" class="${kind}" ${extra}>${s}</text>`;
  const path = (d,kind='p-arrow',extra='') => `<path d="${d}" class="${kind}" ${extra}/>`;
  const rect = (x,y,w,h,kind='p-node',extra='') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" class="${kind}" ${extra}/>`;
  const img = (x,y,w,name,label='') => {
    const h=w*2/3;
    return `<g class="p-thumbnail"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="#101018"/><image x="${x}" y="${y}" width="${w}" height="${h}" href="${asset(name)}" preserveAspectRatio="xMidYMid meet"/>${label?txt(x+w/2,y+h+23,label,'p-image-label','text-anchor="middle"'):''}</g>`;
  };
  let flowIndex=0;
  const flow=(d,stage,kind='') => {
    const id=`pipeline-flow-${flowIndex++}`;
    return `<path id="${id}" d="${d}" class="p-flow ${kind}" data-flow-stage="${stage}"/><circle r="3.5" class="p-flow-packet" data-route="${id}" data-flow-stage="${stage}"/>`;
  };
  const panel=(index,title,content) => {
    const [x,y,w,h]=stages[index].box;
    return `<g class="p-panel" data-panel="${index}" tabindex="0" role="button" aria-label="${stages[index].title}" aria-pressed="false">${rect(x,y,w,h,'p-panel-bg')}${txt(x+18,y+30,`(${String.fromCharCode(97+index)}) ${title}`,'p-panel-title')}${content}</g>`;
  };
  const neuron=(x,y,label) => {
    let out=rect(x,y,99,69,'p-model-box')+txt(x+49.5,y+24,label,'p-model-label','text-anchor="middle"');
    for(let a=0;a<3;a++)for(let b=0;b<2;b++)out+=`<line x1="${x+26}" y1="${y+38+a*10}" x2="${x+72}" y2="${y+43+b*10}" class="p-neuron-link"/>`;
    for(let a=0;a<3;a++)out+=`<circle cx="${x+26}" cy="${y+38+a*10}" r="3" class="p-neuron"/>`;
    for(let b=0;b<2;b++)out+=`<circle cx="${x+72}" cy="${y+43+b*10}" r="3" class="p-neuron"/>`;
    return out;
  };
  const cubes=[];
  for(let layer=0;layer<3;layer++)for(let row=0;row<4;row++)for(let col=0;col<4;col++){
    const x=936+(col-row)*12, y=343+(row+col)*6-layer*14;
    cubes.push(`<g class="p-volume-cell" data-cell="${cubes.length}"><path d="M${x} ${y-7}l11 6-11 6-11-6Z" class="cube-top"/><path d="M${x-11} ${y-1}l11 6v12l-11-6Z" class="cube-left"/><path d="M${x} ${y+5}l11-6v12l-11 6Z" class="cube-right"/></g>`);
  }
  const cells=[];
  const palette=['#c5dcdd','#007d78','#80b6b7','#bf7049','#60959e'];
  for(let row=0;row<9;row++)for(let col=0;col<11;col++){
    const label=col<4?1:row<4?2:col<8?3:4;
    cells.push(`<rect x="${337+col*14}" y="${382+row*14}" width="12" height="12" rx="1.5" fill="${palette[label]}" class="p-region-cell" data-order="${row+Math.abs(col-5)}"/>`);
  }
  const particles=[];
  for(let i=0;i<35;i++){
    const a=i*2.39996, radius=16+((i*19)%58);
    const dx=Math.cos(a)*radius,dy=Math.sin(a)*radius*.47;
    particles.push(`<circle cx="${1175+dx}" cy="${340+dy}" r="${i%3===0?3.6:2.6}" data-dx="${dx}" data-dy="${dy}" class="p-particle ${i%3===0?'p-particle-amber':''}"/>`);
  }
  const rankRows=['Score','Recall','IoU','Edges','Depth','Normals','Reproj.','Centroid'];
  const ranks=[[3,1,2,4],[4,2,1,3],[3,1,2,4],[4,1,3,2],[2,1,3,4],[3,2,1,4],[4,1,2,3],[3,1,2,4]];
  const matrix=ranks.map((row,r)=>txt(1257,560+r*18,rankRows[r],'p-rank-label')+row.map((rank,c)=>`<g class="p-rank-cell" data-row="${r}" data-column="${c}">${rect(1310+c*24,547+r*18,22,17,`p-rank-${rank}`)}${txt(1321+c*24,560+r*18,rank,'p-rank-digit','text-anchor="middle"')}</g>`).join('')).join('');

  host.innerHTML=`<svg class="pipeline-svg" viewBox="0 0 1440 820" aria-label="PlanarCityLoc: five-stage animated pipeline" role="group">
    <defs><marker id="pipeline-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="m1 1 6 3-6 3" fill="none" stroke="#728c94" stroke-width="1.4"/></marker><marker id="pipeline-loop-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="m1 1 6 3-6 3" fill="none" stroke="#007d78" stroke-width="1.4"/></marker></defs>
    <g class="p-shared">
      ${rect(16,16,260,162,'p-input-bg')}${txt(34,44,'Query image Iq','p-panel-title')}${img(79,55,150,'query-rgb')}
      ${rect(294,16,1130,162,'p-renderer-bg')}${txt(314,44,'LoD model M','p-panel-title')}
      <image x="308" y="58" width="205" height="94" href="${asset('lod-model')}" preserveAspectRatio="xMidYMid meet"/>
      ${txt(410,165,'pose hypothesis ξ','p-image-label','text-anchor="middle"')}
      ${flow('M520 99H554',2)}${rect(562,73,112,50,'p-renderer-node')}${txt(618,104,'Renderer','p-model-label','text-anchor="middle"')}
      ${flow('M681 99H710',2)}
      ${img(718,65,103,'rendered-depth','D̂(ξ)')}${img(835,65,103,'rendered-mask','M̂(ξ)')}${img(952,65,103,'rendered-normals','N̂(ξ)')}${img(1069,65,103,'rendered-edges','Ê(ξ)')}
      ${flow('M1180 99H1220',1)}${txt(1202,61,'Extract','p-small','text-anchor="middle"')}${img(1240,55,150,'rendered-planes','3D planes Q(ξ)')}
    </g>
    <g class="p-external">
      ${flow('M115 178V270H85V282',0)}${flow('M181 178V260H211V282',0)}
      ${flow('M276 570H283V300H311',1)}
      ${flow('M1024 178V190H1101V202',2)}
      ${flow('M1315 178V444H788V458',3)}
      ${flow('M1328 430V446H1176V458',3)}
      ${flow('M534 702H558',3)}${flow('M1220 750H1244',4)}
    </g>
    ${panel(0,'Query geometry',`
      ${neuron(35,282,'MoGe-2')}${neuron(159,282,'SAM 3')}
      ${flow('M84 357V383',0)}${flow('M208 357V383',0)}
      <g class="p-query-output" data-output="0">${img(35,391,105,'query-depth','Depth Dq')}</g>
      <g class="p-query-output" data-output="1">${img(159,391,105,'query-mask','Buildings Mq')}</g>
      ${flow('M26 319H23V541H33',0)}
      <g class="p-query-output" data-output="2">${img(35,523,105,'query-normals','Normals Nq')}</g>
      <g class="p-query-output" data-output="3">${img(159,655,105,'query-edges','Edges Eq')}</g>
      ${flow('M89 626V640H181V648',0)}
      ${rect(137,616,86,27,'p-operator')}${txt(180,635,'Sobel','p-small','text-anchor="middle"')}
      ${flow('M211 489V608',0)}${txt(193,571,'mask','p-small')}
      ${txt(34,775,'Zero-shot predictions','p-small')}
    `)}
    ${panel(1,'Plane extraction',`
      ${img(313,277,62,'query-normals','Nq')}${img(383,277,62,'query-mask','Mq')}${img(453,277,62,'query-depth','Dq')}
      ${flow('M413 350V373',1)}
      ${cells.join('')}${rect(331,376,166,136,'p-grid-frame')}
      <circle cx="${337+5*14+6}" cy="${382+4*14+6}" r="5" class="p-seed"/>
      ${txt(414,539,'Region growing','p-model-label','text-anchor="middle"')}
      ${txt(414,568,'normal + depth gates','p-small','text-anchor="middle"')}
      ${flow('M414 576V599',1)}${img(314,610,200,'query-planes','2D patches Pq = {Pj}')}
      ${txt(414,781,'centroids p̄j · normals n̄j','p-small','text-anchor="middle"')}
    `)}
    ${panel(2,'Geometry-aware coarse initialization',`
      ${txt(584,280,'GPS prior','p-image-label')}${txt(610,342,'ξp','p-pose-label','text-anchor="middle"')}
      ${txt(820,273,'S(ξ) = wm SM + we SE + wd SD','p-formula')}
      ${flow('M635 340H674',2)}
      ${[22,35,58,80,64,31,18,9].map((h,i)=>`<rect x="${692+i*17}" y="${378-h}" width="11" height="${h}" rx="2" class="p-z-bar" data-z="${i}"/>`).join('')}
      ${path('M684 290V380H836','p-axis')}${txt(757,408,'Z-prescreen','p-image-label','text-anchor="middle"')}
      ${flow('M842 340H875',2)}${cubes.join('')}${txt(950,408,'4D cost volume','p-image-label','text-anchor="middle"')}
      ${flow('M1019 340H1081',2)}${particles.join('')}
      <circle cx="1175" cy="340" r="5" class="p-particle-target"/>
      ${txt(1172,408,'Multi-scale PF','p-image-label','text-anchor="middle"')}
      ${flow('M1258 340H1300',2)}${txt(1340,346,'ξf','p-pose-label','text-anchor="middle"')}${txt(1340,408,'Coarse pose','p-image-label','text-anchor="middle"')}
    `)}
    ${panel(3,'Iterative plane-based PnP',`
      ${rect(578,510,106,35,'p-operator')}${txt(631,534,'Render ξ(t)','p-small','text-anchor="middle"')}
      ${flow('M689 528H743',3)}${txt(773,534,'Q(ξ(t)), D̂(ξ(t))','p-formula')}
      ${path('M1158 604V500H631V507','p-loop')}${txt(1128,529,'re-render','p-small','text-anchor="middle"')}
      ${txt(643,572,'Query planes Pj','p-image-label','text-anchor="middle"')}${img(579,585,128,'query-overlay')}
      ${txt(808,572,'LoD planes Qi','p-image-label','text-anchor="middle"')}${img(744,585,128,'rendered-planes')}
      ${txt(973,572,'Depth → Xi','p-image-label','text-anchor="middle"')}${img(909,585,128,'rendered-depth')}
      <g class="p-correspondences">
        ${path('M623 610L793 617L948 610','p-correspondence')}${path('M668 644L839 649L1000 645','p-correspondence')}
        <g class="p-query-centroids"><circle cx="623" cy="610" r="5"/><circle cx="668" cy="644" r="5"/></g>
        <g class="p-lod-centroids"><circle cx="793" cy="617" r="5"/><circle cx="839" cy="649" r="5"/></g>
        <g class="p-depth-centroids"><circle cx="948" cy="610" r="5"/><circle cx="1000" cy="645" r="5"/></g>
      </g>
      ${flow('M1048 630H1082',3)}
      <g id="pipeline-camera" transform="translate(1134 624)">
        <path d="M0-22-30 25H30ZM-30 25-22 38H22L30 25M0-22-22 38M0-22 22 38" class="p-frustum"/>
        <path d="M-12 0 0-8 14 0 0 10Z" class="p-camera-plane"/>
        <circle cx="0" cy="-22" r="4" class="p-camera-center"/>
      </g>
      ${txt(1138,686,'PnP-RANSAC','p-model-label','text-anchor="middle"')}
      ${txt(579,704,'Overlap voting → centroid consistency','p-small')}
      ${txt(579,730,'(Xi, p̄j) + K → ξ(t+1)','p-formula')}
      ${txt(1140,731,'t = 1','p-small','text-anchor="middle" id="pipeline-iteration"')}
      ${['ξf','ξ(1)','ξ(2)','…','ξ(N)'].map((s,i)=>rect(582+i*102,752,82,29,'p-candidate',`data-candidate="${i}"`)+txt(623+i*102,773,s,'p-small','text-anchor="middle"')).join('')}
    `)}
    ${panel(4,'Final pose',`
      ${txt(1334,517,'Candidate ranks','p-image-label','text-anchor="middle"')}
      ${['ξf','ξ¹','ξ²','ξN'].map((s,i)=>txt(1321+i*24,540,s,'p-rank-label','text-anchor="middle"')).join('')}
      ${matrix}
      <rect x="1334" y="545" width="23" height="145" rx="3" class="p-winning-column"/>
      ${flow('M1344 698V730',4)}${txt(1320,758,'ξ*','p-pose-label','text-anchor="middle"')}
      <path d="m1350 741 7 7 14-17" class="p-winner-check"/>
      ${txt(1334,782,'Rank aggregation','p-small','text-anchor="middle"')}
    `)}
  </svg>`;

  const svg=host.querySelector('svg');
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const narrow=matchMedia('(max-width: 700px)');
  const buttons=[...root.querySelectorAll('[data-pipeline-step]')];
  const panels=[...svg.querySelectorAll('[data-panel]')];
  const playback=root.querySelector('#pipeline-play');
  const slider=root.querySelector('#pipeline-time');
  const focusButton=root.querySelector('#pipeline-focus');
  const queryOutputs=[...svg.querySelectorAll('.p-query-output')];
  const regionCells=[...svg.querySelectorAll('.p-region-cell')];
  const zBars=[...svg.querySelectorAll('.p-z-bar')];
  const volumeCells=[...svg.querySelectorAll('.p-volume-cell')];
  const particleNodes=[...svg.querySelectorAll('.p-particle')];
  const correspondenceNodes=[...svg.querySelectorAll('.p-correspondence')];
  const rankNodes=[...svg.querySelectorAll('.p-rank-cell')];
  const flowPaths=[...svg.querySelectorAll('.p-flow')];
  const packets=[...svg.querySelectorAll('.p-flow-packet')].map(node=>{
    const route=svg.querySelector(`#${node.dataset.route}`);
    return {node,route,length:route.getTotalLength(),stage:Number(node.dataset.flowStage)};
  });
  let elapsed=reduced.matches?4500:0;
  let active=-1;
  let playing=!reduced.matches;
  let focused=narrow.matches;
  let visible=false;
  let request=null;
  let previous=null;
  const clamp=(n)=>Math.min(1,Math.max(0,n));
  const ease=(n)=>n*n*(3-2*n);

  function layout() {
    svg.classList.toggle('is-focused',focused);
    host.classList.toggle('is-focused',focused);
    focusButton.setAttribute('aria-pressed',String(focused));
    focusButton.textContent=focused?'Full pipeline':'Focus stage';
    const [x,y,w,h]=stages[Math.max(0,active)].box;
    svg.setAttribute('viewBox',focused?`${x-10} ${y-10} ${w+20} ${h+20}`:'0 0 1440 820');
    panels.forEach((panel,index)=>{
      panel.setAttribute('tabindex',!focused||index===active?'0':'-1');
      panel.setAttribute('aria-hidden',String(focused&&index!==active));
    });
  }

  function selectStage(index) {
    active=index;
    const detail=stages[index];
    root.dataset.stage=index;
    root.querySelector('#pipeline-position').textContent=`${String.fromCharCode(97+index)} / e`;
    root.querySelector('#pipeline-detail-tag').textContent=detail.tag;
    root.querySelector('#pipeline-detail-title').textContent=detail.title;
    root.querySelector('#pipeline-detail-text').textContent=detail.text;
    root.querySelector('#pipeline-input').textContent=detail.input;
    root.querySelector('#pipeline-output').textContent=detail.output;
    [...buttons,...panels].forEach(node=>{
      const selected=Number(node.dataset.pipelineStep??node.dataset.panel)===index;
      node.setAttribute('aria-pressed',String(selected));
      node.classList.toggle('is-active',selected);
    });
    layout();
  }

  function render() {
    const index=Math.min(4,Math.floor(elapsed/5000));
    if(active!==index)selectStage(index);
    const progress=(elapsed%5000)/5000;
    const phase=(stage)=>stage===active?progress:1;
    queryOutputs.forEach((node,i)=>node.style.opacity=.2+.8*clamp((phase(0)*4-i)*2));
    regionCells.forEach(node=>node.style.opacity=.12+.88*clamp((phase(1)*19-Number(node.dataset.order))*2));
    const coarse=phase(2);
    zBars.forEach((bar,i)=>bar.classList.toggle('is-best',i===(coarse<.3?Math.floor(coarse/.3*8):3)));
    volumeCells.forEach((cell,i)=>cell.classList.toggle('is-selected',i===(coarse<.65?Math.floor(clamp((coarse-.3)/.35)*47):38)));
    const contraction=1-.85*ease(clamp((coarse-.65)/.35));
    particleNodes.forEach(node=>{
      node.setAttribute('cx',1175+Number(node.dataset.dx)*contraction);
      node.setAttribute('cy',340+Number(node.dataset.dy)*contraction);
    });
    const refinement=phase(3);
    const iteration=Math.min(3,Math.floor(refinement*3)+1);
    const residual=1-ease(refinement);
    correspondenceNodes.forEach(node=>{node.style.opacity=.25+.75*clamp(refinement*5);node.style.strokeDashoffset=-refinement*40;});
    svg.querySelector('.p-lod-centroids').setAttribute('transform',`translate(${12*residual} ${-6*residual})`);
    correspondenceNodes[0].setAttribute('d',`M623 610L${793+12*residual} ${617-6*residual}L948 610`);
    correspondenceNodes[1].setAttribute('d',`M668 644L${839+12*residual} ${649-6*residual}L1000 645`);
    svg.querySelector('#pipeline-camera').setAttribute('transform',`translate(${1134+12*residual} ${624-9*residual}) rotate(${-12*residual})`);
    svg.querySelector('#pipeline-iteration').textContent=`t = ${iteration}`;
    svg.querySelectorAll('.p-candidate').forEach((node,i)=>node.classList.toggle('is-ready',i===0||i<=iteration));
    rankNodes.forEach(node=>node.style.opacity=.2+.8*clamp(phase(4)*10-Number(node.dataset.row)));
    svg.querySelector('.p-winning-column').style.opacity=clamp((phase(4)-.8)*5);
    svg.querySelector('.p-winner-check').style.opacity=clamp((phase(4)-.8)*5);
    flowPaths.forEach(node=>node.classList.toggle('is-flowing',Number(node.dataset.flowStage)===active));
    packets.forEach(({node,route,length,stage},i)=>{
      node.style.opacity=stage===active?1:0;
      if(stage===active){const point=route.getPointAtLength(((progress*2+i*.17)%1)*length);node.setAttribute('cx',point.x);node.setAttribute('cy',point.y);}
    });
    slider.value=Math.floor(elapsed);
    root.querySelector('#pipeline-time-label').textContent=`0:${String(Math.floor(elapsed/1000)).padStart(2,'0')} / 0:25`;
  }

  function tick(now) {
    if(previous!==null)elapsed=(elapsed+Math.min(100,now-previous))%25000;
    previous=now;
    render();
    request=requestAnimationFrame(tick);
  }

  function sync() {
    if(request!==null)cancelAnimationFrame(request);
    request=null;
    previous=null;
    root.dataset.playing=String(playing);
    playback.querySelector('span').textContent=playing?'Pause':'Play';
    playback.setAttribute('aria-label',playing?'Pause pipeline animation':'Play pipeline animation');
    if(playing&&visible&&!document.hidden)request=requestAnimationFrame(tick);
  }

  function jump(index) { elapsed=index*5000+4500;playing=false;render();sync(); }
  buttons.forEach((button,index)=>{
    button.addEventListener('click',()=>jump(index));
    button.addEventListener('keydown',event=>{
      const next=event.key==='ArrowRight'?(index+1)%5:event.key==='ArrowLeft'?(index+4)%5:event.key==='Home'?0:event.key==='End'?4:null;
      if(next!==null){event.preventDefault();buttons[next].focus();jump(next);}
    });
  });
  panels.forEach((panel,index)=>{
    panel.addEventListener('click',()=>jump(index));
    panel.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();jump(index);}});
  });
  playback.addEventListener('click',()=>{playing=!playing;sync();});
  root.querySelector('#pipeline-replay').addEventListener('click',()=>{elapsed=0;playing=true;render();sync();});
  slider.addEventListener('input',()=>{elapsed=Number(slider.value);playing=false;render();sync();});
  focusButton.addEventListener('click',()=>{focused=!focused;layout();});
  new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;sync();},{threshold:.08}).observe(host);
  document.addEventListener('visibilitychange',sync);
  reduced.addEventListener('change',()=>{playing=false;elapsed=active*5000+4500;render();sync();});
  narrow.addEventListener('change',()=>{focused=narrow.matches;layout();});
  render();
  sync();
})();
