// Beamer-style PowerPoint companion. Scientific series and tables remain editable.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
const root=process.env.PROJECT_ROOT??path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime='C:/Users/14108/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const skill=process.env.PRESENTATION_SKILL_DIR??'D:/CodexData/.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.11052/skills/presentations';
process.env.RUNTIME_NODE_MODULES??=runtime+'/node/node_modules';
const materials=process.env.MATERIALS_DIR??'reports/project2_rsrs';
const evidence=process.env.EVIDENCE_DIR??'evidence/project2_rsrs';
const build=path.join(root,'tmp',process.env.BUILD_NAME??'ppt-project2-rsrs');
const final=path.join(root,materials,process.env.PPTX_NAME??'CF2026_Project2_Presentation.pptx');
const data=JSON.parse(await fs.readFile(path.join(root,materials,'deck_content.json'),'utf8'));
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const p=Presentation.create({slideSize:{width:1280,height:720}});
const navy='#142B43',teal='#168579',gray='#627285',family='Microsoft YaHei';
const colors=[teal,'#4669A1','#B37A50','#8A9099'];
function box(s,x,y,w,h,fill){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});}
function text(s,value,x,y,w,h,size=27,color=navy,bold=false){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=String(value);t.text.style={typeface:family,fontSize:size,color,bold,autoFit:'none'};return t;
}
function items(s,list,y=241,size=29,step=88){list.forEach((a,k)=>{text(s,String(k+1).padStart(2,'0'),61,y+k*step,55,48,size,teal,true);text(s,a,125,y+k*step,1090,step-12,size);});}
async function csv(relative){
 const raw=await fs.readFile(path.join(root,evidence,relative),'utf8');
 const rows=[];let row=[],cell='',quoted=false;
 for(let i=0;i<raw.length;i++){
  const c=raw[i];if(c==='"'){if(quoted&&raw[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}
  else if(c===','&&!quoted){row.push(cell);cell='';}
  else if(c==='\n'&&!quoted){row.push(cell.replace(/\r$/,''));if(row.some(x=>x))rows.push(row);row=[];cell='';}
  else cell+=c;
 }
 if(cell||row.length){row.push(cell);rows.push(row);}const heads=rows.shift();
 return rows.map(r=>Object.fromEntries(heads.map((h,k)=>[h,r[k]])));
}
function downsample(rows,max=140){if(rows.length<=max)return rows;const step=Math.ceil(rows.length/max);return rows.filter((_,i)=>i%step===0||i===rows.length-1);}
async function rsrsComparison(figure){
 const selected=[];
 if(figure==='reproduction_curves.png'){
  const table=await csv('source_compatible.csv');
  for(const [id,name] of [['standard_ma','标准分+MA'],['corrected_ma','修正分+MA'],['right_ma','右偏分+MA']]){
   const row=table.find(x=>x.strategy_id===id&&x.warmup==='partial_prior_M_min2'&&x.end_assumption==='march_end'&&+x.cost_bps===0);
   selected.push({name,rows:downsample(await csv(row.daily_file))});
  }
 }else{
  const table=await csv('diagnostics.csv');
  for(const cost of [0,10,20,30]){
   const row=table.find(x=>x.period_id==='historical_overlap'&&+x.cost_bps===cost&&(x.case===`cost_${cost}bps`||(cost===10&&x.case.startsWith('prior_M'))));
   selected.push({name:`每边${cost}bp`,rows:downsample(await csv(row.daily_file))});
  }
 }
 const series=selected.map(s=>({name:s.name,values:s.rows.map(r=>+r.nav)}));
 if(figure==='reproduction_curves.png')series.push({name:'买入持有',values:selected[0].rows.map(r=>+r.benchmark_nav)});
 return {type:'line',categories:selected[0].rows.map(r=>r.date.slice(0,10)),series,axisFormat:'0.0'};
}
const chartOwners=[],tableOwners=[];
function chart(s,c,pos,slideNo,small=false){
 const ch=s.charts.add(c.type??'line',{position:{left:pos[0],top:pos[1],width:pos[2],height:pos[3]},categories:c.categories,
  series:c.series.map((v,k)=>({name:v.name,values:v.values.map(x=>+Number(x).toFixed(8)),fill:colors[k%colors.length],line:{fill:colors[k%colors.length],width:2.1},marker:{symbol:'none'},valuesFormatCode:c.axisFormat??'0.00'})),
  hasLegend:c.series.length>1,legend:{position:'top',textStyle:{fontSize:small?15:20}},
  barOptions:{direction:c.horizontal?'bar':'column',grouping:'clustered',gapWidth:90},lineOptions:{smooth:false},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',
  xAxis:{textStyle:{fontSize:small?14:18},tickLabelPosition:c.type==='line'?'none':'low',line:{fill:'#C8D2DD',width:1}},
  yAxis:{textStyle:{fontSize:small?15:20},numberFormatCode:c.axisFormat??'0.00',majorGridlines:{fill:'#DEE6EE',width:1}},
  dataLabels:{showValue:false}});
 applyPresentationChartFont(ch,{fontFamily:family});chartOwners.push(slideNo);
 if(c.type==='line'){text(s,c.categories[0],pos[0]+35,pos[1]+pos[3]-5,130,27,small?15:18,gray);text(s,c.categories.at(-1),pos[0]+pos[2]-128,pos[1]+pos[3]-5,130,27,small?15:18,gray);}
}
async function addImage(s,relative,pos,alt){const bytes=await fs.readFile(path.join(root,relative));s.images.add({blob:new Uint8Array(bytes),contentType:'image/png',alt,fit:'contain',position:{left:pos[0],top:pos[1],width:pos[2],height:pos[3]}});}
function table(s,config,pos,slideNo){
 const values=Array.isArray(config)?config:[config.headers,...config.rows];const n=values[0].length;
 const widths=config.columnWidths??(n===2?[pos[2]*.32,pos[2]*.68]:[pos[2]*.34,...Array(n-1).fill(pos[2]*.66/(n-1))]);
 const tb=s.tables.add({rows:values.length,columns:n,left:pos[0],top:pos[1],width:pos[2],height:pos[3],values,columnWidths:widths});
 tb.borders.assign({fill:'#D9E3EA',width:1});
 for(let r=0;r<values.length;r++)for(let c=0;c<n;c++){const cell=tb.getCell(r,c);cell.fill=r===0?navy:(r%2?'#F0F5F8':'#FFFFFF');cell.text.style={typeface:family,fontSize:n>4?21:24,color:r===0?'#FFFFFF':navy,bold:r===0,autoFit:'none'};}
 tableOwners.push(slideNo);
}
for(let i=0;i<data.length;i++){
 const d=data[i],s=p.slides.add();s.background.fill='#FFFFFF';
 box(s,0,0,1280,92,navy);text(s,d.title,52,21,1180,58,36,'#FFFFFF',true);
 box(s,0,686,1280,34,navy);text(s,'青序量化研究平台  ·  CF2026 Project 2',48,692,1050,23,16,'#FFFFFF');text(s,`${i+1} / ${data.length}`,1150,692,100,23,16,'#FFFFFF');
 text(s,d.claim,62,120,1155,84,d.claim.length>40?27:31,navy,true);
 if(d.kind==='cover'){
  text(s,d.cover_title??'研报复现\n与风险改进',70,240,1120,170,62,navy,true);
  text(s,d.items.join('\n'),75,453,1130,151,27,gray);text(s,'14页正文 · 15分钟 · 附可复现实验与逐页讲稿',75,632,1130,35,23,teal);
 }else if(d.kind==='comparison'&&d.original_image){
  text(s,'原文',65,218,500,34,23,teal,true);text(s,'本次实验（口径见图注）',660,218,550,34,23,teal,true);
  await addImage(s,path.join(evidence,'reference',d.original_image),[58,259,565,315],'原文图表，保留来源与图例');
  const c=d.chart??(evidence.endsWith('project2_rsrs')?await rsrsComparison(d.figure):null);
  if(c)chart(s,c,[656,247,565,321],i+1,true);else await addImage(s,path.join(materials,'figures',d.figure),[655,259,565,315],d.title);
  text(s,d.items.join('\n'),65,593,1150,83,22,gray);
 }else if(d.table){
  const vals=Array.isArray(d.table)?d.table:[d.table.headers,...d.table.rows];const h=Math.min(335,vals.length*62);
  table(s,d.table,[64,224,1152,h],i+1);text(s,d.items.join('\n'),66,244+h,1148,128,23,gray);
 }else if(d.chart){chart(s,d.chart,[65,217,1150,354],i+1);text(s,d.items.join('\n'),66,600,1148,76,22,gray);
 }else if(d.figure==='robustness.png'&&evidence.endsWith('project2_rsrs')){
  const rows=(await csv('paired_comparison.csv')).filter(x=>+x.cost_bps===10);
  const c={type:'bar',categories:rows.map(x=>`${x.index_name}/${x.period_name.replace('时期','').replace('历史重叠期','重叠')}`),series:[{name:'CAGR变化（百分点）',values:rows.map(x=>+x.delta_cagr*100)},{name:'回撤变化（百分点，负值为下降）',values:rows.map(x=>+x.delta_max_drawdown*100)}],axisFormat:'0.0'};
  chart(s,c,[65,216,1150,345],i+1);text(s,d.items.join('\n'),67,592,1146,78,21,gray);
 }else if(d.figure){await addImage(s,path.join(materials,'figures',d.figure),[63,218,1155,342],d.title);text(s,d.items.join('\n'),66,590,1148,83,22,gray);
 }else if(d.formula_plain){text(s,d.formula_plain,69,229,1134,87,31,teal,true);items(s,d.items,351,27,91);
 }else{items(s,d.items,242,d.items.length>3?26:29,d.items.length>3?88:105);}
 s.speakerNotes.textFrame.setText(`建议用时：${d.seconds}秒\n${d.notes}\n${d.source??''}`);
}
await fs.mkdir(build,{recursive:true});
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
await fs.writeFile(path.join(build,'deck.proto.json'),JSON.stringify(p.toProto()));
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,pythonExecutable:runtime+'/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 explicitTotalSlideCount:data.length,requiredNativeChartOwnerSlides:chartOwners,requiredNativeTableOwnerSlides:tableOwners,materializeLiteralChartWorkbooks:true,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
 fontPolicy:{basis:'design',families:[family]},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify({final,slides:data.length,chartOwners,tableOwners,validated:true}));
const imported=await PresentationFile.importPptx(await FileBlob.load(final));
const renders=path.join(build,'renders');await fs.mkdir(renders,{recursive:true});
for(let i=0;i<data.length;i++){const blob=await imported.export({slide:imported.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(renders,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));console.log(`Rendered ${i+1}/${data.length}`);}
