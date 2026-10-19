/* Benchmark tables and archived predictions; no simulated scores or poses. */
(() => {
  const data = window.PLANARCITYLOC_RESULTS;
  if (!data) return;
  const $ = selector => document.querySelector(selector);
  // Select successes only for the labeled gallery. Never filter aggregate results.
  const gallery = data.samples.filter(item => {
    const methods = (item.methods ?? ['v2','v3','ours','gt']).filter(method => method !== 'ours' && method !== 'gt');
    return methods.length > 0 && methods.every(method => [0,1].every(metric =>
      Number.isFinite(item.errors.ours?.[metric]) && Number.isFinite(item.errors[method]?.[metric]) && item.errors.ours[metric] < item.errors[method][metric]));
  });
  const datasetButtons = [...document.querySelectorAll('#results [data-dataset]')];
  const sortButtons = [...document.querySelectorAll('[data-sort]')];
  const splitSelect = $('#results-split'), table = $('.results-table');
  const paperOrderButton = $('#results-paper-order');
  const paperSections = [
    {id:'prior'},
    {id:'textured', title:'Textured Mesh-based Models', level:'category'},
    {id:'trained', parent:'LoD-based Models', title:'Requires Training', level:'setup'},
    {id:'oracle', title:'Oracle (GT masks)', level:'setup'},
    {id:'off-the-shelf', title:'Off-the-shelf', level:'setup'},
    {id:'planarcityloc', title:'PlanarCityLoc (Ours)', level:'method'}
  ];
  const sectionFor = row => row.rankingGroup === 'off-the-shelf' && row.method.startsWith('PlanarCityLoc')
    ? 'planarcityloc' : row.rankingGroup ?? 'prior';
  const paperPosition = new Map(data.rows.map((row,index)=>[row.id,index]));
  const tips = {
    method: 'Methods use different map representations; see the map label below each method.',
    model: 'Vision model used by this configuration.',
    setup: 'Setups use different map and training assumptions; compare these alongside the scores.',
    0: 'Recall (%): poses within 2 m translation and 2° rotation error. Higher is better (↑).',
    1: 'Recall (%): poses within 3 m translation and 3° rotation error. Higher is better (↑).',
    2: 'Recall (%): poses within 5 m translation and 5° rotation error. Higher is better (↑).',
    3: 'Median translation error in metres. Lower is better (↓).',
    4: 'Median rotation error in degrees. Lower is better (↓).',
    '†': 'Adapted re-run. Map, rendering, initialization, and search assumptions differ from the published evaluation; see the paper for protocol details.',
    '*': 'Trained on the non-public InsLoD-Loc dataset.'
  };
  const tooltip = document.createElement('div'), descriptions = document.createElement('div'), descriptionIds = new Map();
  tooltip.className = 'table-tooltip'; tooltip.hidden = true; tooltip.setAttribute('aria-hidden','true');
  descriptions.hidden = true; document.body.append(tooltip, descriptions);
  let activeTip = null, tipTimer;
  function addTip(element, text) {
    element.dataset.tableTip = text;
    if (!element.matches('button')) element.tabIndex = 0;
    if (!descriptionIds.has(text)) {
      const description = document.createElement('span'); description.id = `table-tip-${descriptionIds.size}`; description.textContent = text;
      descriptions.append(description); descriptionIds.set(text, description.id);
    }
    element.setAttribute('aria-describedby', descriptionIds.get(text));
    return element;
  }
  function hideTip() { clearTimeout(tipTimer); tooltip.hidden = true; activeTip = null; }
  function showTip(element) {
    clearTimeout(tipTimer); activeTip = element; tooltip.textContent = element.dataset.tableTip;
    // Clear the previous desktop position before measuring on a narrower viewport.
    tooltip.style.left = '0px'; tooltip.style.top = '0px'; tooltip.hidden = false;
    const anchor = element.getBoundingClientRect(), box = tooltip.getBoundingClientRect(), gap = 8;
    const left = Math.max(12, Math.min(innerWidth-box.width-12, anchor.left+(anchor.width-box.width)/2));
    const preferredTop = anchor.bottom+gap+box.height<=innerHeight-12 ? anchor.bottom+gap : anchor.top-box.height-gap;
    const top = Math.max(12, Math.min(innerHeight-box.height-12, preferredTop));
    tooltip.style.left = `${left}px`; tooltip.style.top = `${top}px`;
  }
  const tipTarget = event => event.target instanceof Element ? event.target.closest('#results [data-table-tip]') : null;
  document.addEventListener('pointerover', event => { const target=tipTarget(event); if(target && event.pointerType!=='touch')showTip(target); });
  document.addEventListener('pointerout', event => {
    const target=tipTarget(event);
    if(target && target===activeTip && document.activeElement!==target && !target.contains(event.relatedTarget) && !tooltip.contains(event.relatedTarget))tipTimer=setTimeout(hideTip,140);
  });
  tooltip.addEventListener('pointerenter',()=>clearTimeout(tipTimer));
  tooltip.addEventListener('pointerleave',()=>{if(document.activeElement!==activeTip)hideTip();});
  document.addEventListener('focusin', event => { const target=tipTarget(event); if(target)showTip(target);else hideTip(); });
  document.addEventListener('focusout', event => {if(tipTarget(event))hideTip();});
  document.addEventListener('click', event => {const target=tipTarget(event); if(target)showTip(target);else if(!tooltip.contains(event.target))hideTip();});
  document.addEventListener('keydown', event => {if(event.key==='Escape')hideTip();});
  document.addEventListener('scroll',hideTip,true); window.addEventListener('resize',hideTip);
  sortButtons.forEach(button=>addTip(button,tips[button.dataset.sort]));
  function appendLabel(element, text) {
    text.split(/([†*])/).forEach(part=>{
      if(tips[part] && (part==='†'||part==='*')) {
        const marker=document.createElement('span');marker.className='table-footnote';marker.textContent=part;addTip(marker,tips[part]);element.append(marker);
      } else element.append(document.createTextNode(part));
    });
  }
  let dataset = data.datasets[0], split = 0, sortKey = null, ascending = true;
  const imagePath = (id, method) => `assets/comparisons/${id}-${method}.jpg`;
  const imageCache = new Map();
  function loadImage(url) {
    if (!imageCache.has(url)) imageCache.set(url, new Promise((resolve,reject) => {
      const image = new Image();image.onload = () => resolve(image);
      image.onerror = () => {imageCache.delete(url);reject(Error('Image unavailable'));};image.src=url;
    }));
    return imageCache.get(url);
  }
  function tableValue(row,key) {return /^\d$/.test(key) ? row.metrics[dataset.splits[split].index][Number(key)] : row[key];}
  function renderTable() {
    hideTip();
    const ranksByGroup = new Map();
    for (const group of new Set(data.rows.map(row=>row.rankingGroup).filter(Boolean))) {
      ranksByGroup.set(group, Array.from({length:5},(_,metric) => [...new Set(data.rows
        .filter(row=>row.rankingGroup===group)
        .map(row=>row.metrics[dataset.splits[split].index][metric])
        .filter(Number.isFinite))].sort((a,b)=>metric<3?b-a:a-b).slice(0,2)));
    }
    const compareRows = (a,b) => {
      const paperOrder = paperPosition.get(a.id)-paperPosition.get(b.id);
      if(sortKey===null)return paperOrder;
      const first=tableValue(a,sortKey), second=tableValue(b,sortKey);
      const firstMissing=first===null||first===undefined||(typeof first==='number'&&!Number.isFinite(first));
      const secondMissing=second===null||second===undefined||(typeof second==='number'&&!Number.isFinite(second));
      if(firstMissing!==secondMissing)return firstMissing?1:-1;
      if(firstMissing)return paperOrder;
      return (typeof first==='number'?first-second:first.localeCompare(second))*(ascending?1:-1)||paperOrder;
    };
    const makeRow = row => {
      const tr=document.createElement('tr');tr.dataset.rowId=row.id;if(row.ours)tr.className='ours';
      const name=document.createElement('th');name.scope='row';appendLabel(name,row.method);
      const map=document.createElement('span');map.className='method-note';map.textContent=row.map;addTip(map,`${row.map}. ${tips.method}`);name.append(map);tr.append(name);
      for(const [value,className] of [[row.model,'model-cell'],[row.setup,'setup-cell']]){
        const td=document.createElement('td');td.className=className;
        if(className==='setup-cell' && value!=='—') {
          const label=document.createElement('span');label.textContent=value.replace('*','');addTip(label,tips.setup);td.append(label);
          if(value.includes('*'))appendLabel(td,'*');
        } else appendLabel(td,value);
        tr.append(td);
      }
      row.metrics[dataset.splits[split].index].forEach((value,index)=>{
        const td=document.createElement('td');td.className='metric';
        if(!Number.isFinite(value)){td.textContent='—';tr.append(td);return;}
        const rank=row.rankingGroup?ranksByGroup.get(row.rankingGroup)[index].indexOf(value):-1,text=`${value.toFixed(2)}${index<3?'%':''}`;
        if(rank===0||rank===1){
          const mark=document.createElement(rank===0?'strong':'u');mark.textContent=text;
          addTip(mark,rank===0?'Best value in this setup group. Bold marks all ties.':'Second-best distinct value in this setup group. Underlining marks all ties.');td.append(mark);
        }else td.textContent=text;
        tr.append(td);
      });return tr;
    };
    const makeHeading = (text,level,id) => {
      const tr=document.createElement('tr');tr.className=`table-group table-group-${level}`;
      const th=document.createElement('th');th.colSpan=8;th.textContent=text;
      if(id){th.id=id;th.scope='rowgroup';}
      tr.append(th);return tr;
    };
    const groups=paperSections.map(section=>{
      const body=document.createElement('tbody');body.dataset.resultGroup=section.id;
      if(section.parent)body.append(makeHeading(section.parent,'category'));
      if(section.title){
        const id=`results-group-${section.id}`;
        body.append(makeHeading(section.title,section.level,id));body.setAttribute('aria-labelledby',id);
      }
      body.append(...data.rows.filter(row=>sectionFor(row)===section.id).sort(compareRows).map(makeRow));
      return body;
    });
    [...table.tBodies].forEach(body=>body.remove());table.append(...groups);
    paperOrderButton.disabled=sortKey===null;
    $('#results-order-label').textContent=sortKey===null?'Paper order':'Custom order within groups';
    $('#results-caption').textContent=`${dataset.name} · ${dataset.splits[split].name} · ${dataset.lod}`;
    sortButtons.forEach(button=>{
      const selected=button.dataset.sort===sortKey;
      button.closest('th').setAttribute('aria-sort',selected?(ascending?'ascending':'descending'):'none');
      button.querySelector('span').textContent=selected?(ascending?'↑':'↓'):'↕';
    });
  }
  function selectDataset(id) {
    dataset=data.datasets.find(item=>item.id===id);split=0;
    datasetButtons.forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.dataset===id)));
    splitSelect.replaceChildren(...dataset.splits.map((item,index)=>new Option(item.name,String(index))));renderTable();
  }
  datasetButtons.forEach(button=>button.addEventListener('click',()=>selectDataset(button.dataset.dataset)));
  splitSelect.addEventListener('change',()=>{split=Number(splitSelect.value);renderTable();});
  sortButtons.forEach(button=>button.addEventListener('click',()=>{
    const key=button.dataset.sort,bestFirst=!['0','1','2'].includes(key);
    if(key!==sortKey){sortKey=key;ascending=bestFirst;}else if(ascending===bestFirst)ascending=!bestFirst;else sortKey=null;
    renderTable();$('#results-sort-status').textContent=sortKey===null?'Paper order restored.':`Sorted within each displayed group by ${button.textContent.replace(/[↕↑↓]/g,'').trim()}, ${ascending?'ascending':'descending'}.`;
  }));
  paperOrderButton.addEventListener('click',()=>{
    sortKey=null;renderTable();$('#results-sort-status').textContent='Paper order restored.';
  });

  function createComparison(ds, group) {
    const samples=gallery.filter(item=>item.dataset===ds.id&&item.split===group.name).sort((a,b)=>Number(Boolean(b.origin))-Number(Boolean(a.origin)));
    if(!samples.length)return;
    const card=$('#comparison-card-template').content.firstElementChild.cloneNode(true);
    const field=name=>card.querySelector(`[data-field="${name}"]`), action=name=>card.querySelector(`[data-action="${name}"]`);
    card.dataset.dataset=ds.id;card.dataset.viewer=group.sample;
    field('dataset').textContent=ds.name;field('split').textContent=group.name;field('lod').textContent=ds.lod;
    const titleId=`${group.sample}-title`;field('split').id=titleId;card.setAttribute('aria-labelledby',titleId);
    const left=field('left'),right=field('right'),box=field('images'),slider=field('slider'),status=field('status'),tabs=field('samples');
    let position=0,version=0,dragging=false;
    [left,right].forEach(select=>select.replaceChildren(...data.methods.map(method=>new Option(method.label,method.id))));left.value='v2';right.value='ours';
    function updateMethods(sample) {
      const allowed=sample.methods??['v2','v3','ours','gt'];
      const available=data.methods.filter(method=>allowed.includes(method.id));
      [left,right].forEach((select,index)=>{
        const previous=select.value;
        select.replaceChildren(...available.map(method=>new Option(method.label,method.id)));
        select.value=available.some(method=>method.id===previous)?previous:
          (index===1&&available.some(method=>method.id==='ours')?'ours':available[0]?.id??'');
      });
    }
    function divider() {
      const value=Number(slider.value);box.style.setProperty('--split',`${value}%`);
      slider.setAttribute('aria-valuetext',`${value}% left method, ${100-value}% right method`);
      field('left-label').style.opacity=value<15?'0':'1';field('right-label').style.opacity=value>85?'0':'1';
    }
    async function renderPair() {
      const current=++version,sample=samples[position],methods=[left,right].map(select=>data.methods.find(method=>method.id===select.value));
      const urls=methods.map(method=>imagePath(sample.id,method.id));box.setAttribute('aria-busy','true');status.textContent='Loading comparison…';
      try {
        await Promise.all(urls.map(loadImage));if(current!==version)return;
        box.style.aspectRatio=`${sample.width} / ${sample.height}`;
        methods.forEach((method,index)=>{
          const side=index?'right':'left',image=field(`${side}-image`),error=sample.errors[method.id];
          image.src=urls[index];image.width=sample.width;image.height=sample.height;image.alt=`${method.label}, ${ds.name} ${group.name}, ${sample.frame||sample.id}`;
          field(`${side}-label`).textContent=method.name;
          const label=document.createElement('strong');label.textContent=method.name;
          field(`${side}-error`).replaceChildren(label,document.createElement('br'),error?`${error[0].toFixed(2)} m · ${error[1].toFixed(2)}°`:'Ground-truth reference');
        });
        card.dataset.sample=sample.id;field('meta').textContent=sample.frame||sample.id;status.textContent='';
      } catch {if(current===version)status.textContent='Could not load this pair. Select a method again to retry.';}
      finally {if(current===version)box.setAttribute('aria-busy','false');}
    }
    function keepThumbnailVisible() {
      const selected=tabs.children[position];if(selected){const a=selected.getBoundingClientRect(),b=tabs.getBoundingClientRect();tabs.scrollLeft+=a.left-b.left-(b.width-a.width)/2;}
    }
    function select(index) {
      position=Math.max(0,Math.min(samples.length-1,index));updateMethods(samples[position]);field('count').textContent=`Example ${position+1} of ${samples.length}`;
      action('previous').disabled=position===0;action('next').disabled=position===samples.length-1;
      [...tabs.children].forEach((button,i)=>button.setAttribute('aria-pressed',String(i===position)));
      keepThumbnailVisible();
      renderPair();
    }
    samples.forEach((sample,index)=>{
      const button=document.createElement('button');button.type='button';button.dataset.sample=sample.id;
      button.setAttribute('aria-label',`${ds.name} ${group.name}, example ${index+1}`);
      const image=document.createElement('img');image.src=imagePath(sample.id,'gt');image.alt='';image.width=80;image.height=56;image.loading='lazy';
      button.append(image,String(index+1));button.addEventListener('click',()=>select(index));
      button.addEventListener('keydown',event=>{
        const next=event.key==='ArrowRight'?Math.min(index+1,samples.length-1):event.key==='ArrowLeft'?Math.max(index-1,0):event.key==='Home'?0:event.key==='End'?samples.length-1:null;
        if(next!==null){event.preventDefault();tabs.children[next].focus();select(next);}
      });tabs.append(button);
    });
    [left,right].forEach(select=>select.addEventListener('change',renderPair));
    action('swap').addEventListener('click',()=>{[left.value,right.value]=[right.value,left.value];renderPair();});
    action('previous').addEventListener('click',()=>select(position-1));action('next').addEventListener('click',()=>select(position+1));
    slider.addEventListener('input',divider);
    function drag(event){const r=box.getBoundingClientRect();slider.value=Math.round(Math.max(0,Math.min(100,(event.clientX-r.left)/r.width*100)));divider();}
    slider.addEventListener('pointerdown',event=>{if(event.button!==0)return;event.preventDefault();slider.focus({preventScroll:true});slider.setPointerCapture(event.pointerId);dragging=true;drag(event);});
    slider.addEventListener('pointermove',event=>{if(dragging)drag(event);});
    slider.addEventListener('pointerup',()=>dragging=false);slider.addEventListener('pointercancel',()=>dragging=false);
    $('#comparison-grid').append(card);select(0);divider();new ResizeObserver(keepThumbnailVisible).observe(tabs);
  }
  data.datasets.forEach(ds=>ds.splits.forEach(group=>createComparison(ds,group)));
  const galleryButtons=[...document.querySelectorAll('[data-gallery-dataset]')];
  galleryButtons.forEach(button=>button.addEventListener('click',()=>{
    const id=button.dataset.galleryDataset;galleryButtons.forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
    document.querySelectorAll('#comparison-grid > article').forEach(card=>card.hidden=id!=='all'&&card.dataset.dataset!==id);
  }));
  selectDataset('uav');
})();
