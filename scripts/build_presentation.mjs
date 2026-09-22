// Execute a copy from tmp/ppt-build, linked to the supplied Artifact Tool runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const root=process.env.PROJECT_ROOT ?? 'D:/Desktop/cf2026-quant-platform';
const skill='D:/CodexData/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
const python='C:/Users/14108/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/14108/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const build=path.join(root,'tmp/ppt-build');
const materials=process.env.MATERIALS_DIR ?? 'reports/final';
const final=path.join(root,materials,process.env.PPTX_NAME ?? 'CF2026_Presentation.pptx');
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const data=JSON.parse(await fs.readFile(path.join(root,materials,'deck_content.json'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const family='Microsoft YaHei',navy='#142B43',teal='#168579',gray='#627285';
const palette=[teal,'#4265A6','#8C99A6','#BB7957'];
function box(s,x,y,w,h,fill){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});}
function text(s,str,x,y,w,h,size=28,color=navy,bold=false){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=str;t.text.style={typeface:family,fontSize:size,color,bold,autoFit:'none'};return t;
}
function items(s,list,y=238,size=29,step=82){
 list.forEach((a,j)=>{box(s,61,y+j*step+12,7,7,teal);text(s,a,87,y+j*step,1108,step-10,size);});
}
const chartOwners=[],tableOwners=[];
for(let j=0;j<data.length;j++){
 const d=data[j],s=p.slides.add();s.background.fill='#FFFFFF';
 box(s,0,0,1280,93,navy);text(s,d.title,54,20,1175,62,38,'#FFFFFF',true);
 box(s,0,685,1280,35,navy);text(s,'青序量化研究平台   CF2026 Project 1',48,690,950,24,16,'#FFFFFF');
 text(s,`${j+1} / ${data.length}`,1150,690,100,24,16,'#FFFFFF');
 box(s,50,119,1180,72,'#EDF5F4');box(s,50,119,7,72,teal);text(s,d.claim,75,133,1124,47,30,navy,true);
 if(d.kind==='cover'){
   text(s,'可复现的多因子研究\n与风险控制',73,247,1110,175,58,navy,true);
   text(s,d.items.join('\n'),78,459,1120,135,27,gray);
   text(s,(d.date??'2026年9月21日')+'   /   20分钟课程展示',78,623,1100,36,24,teal);
 }else if(d.chart){
   const c=d.figure==='factor_ic'?{...d.chart,horizontal:false,categories:['Mom','Rev','Low vol','Range','Trend','Volume','Illiq','Turnover','E/P','B/P','Div','ROE']}:d.chart;const hasItems=d.items.length>0;
   const height=c.horizontal?398:(c.type==='line'?320:(hasItems?338:414));
   const chart=s.charts.add(c.type,{position:{left:64,top:213,width:1150,height},categories:c.categories,
    series:c.series.map((v,k)=>({name:v.name,values:v.values.map(x=>Number(x.toFixed(8))),fill:palette[k%palette.length],line:{fill:palette[k%palette.length],width:2.3},marker:{symbol:'none'},valuesFormatCode:c.percent_points?'0.00':'0.0000'})),
    hasLegend:c.series.length>1,legend:{position:'top',textStyle:{fontSize:20}},
    barOptions:{direction:c.horizontal?'bar':'column',grouping:'clustered',gapWidth:80},
    lineOptions:{smooth:false},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',
    xAxis:{textStyle:{fontSize:c.type==='line'||d.figure==='factor_ic'?17:23},tickLabelPosition:c.type==='line'?'none':'low',line:{fill:'#BCC9D3',width:1}},
    yAxis:{textStyle:{fontSize:20},tickLabelPosition:'low',min:c.type==='line'?.85:undefined,max:c.type==='line'?1.35:undefined,majorUnit:c.type==='line'?.1:undefined,numberFormatCode:c.percent_points?'0.0"%"':'0.00',majorGridlines:{fill:'#DFE6EC',width:1}},
    dataLabels:{showValue:c.type==='bar'&&!c.horizontal&&d.figure!=='factor_ic'&&c.categories.length<=7,position:'outEnd',textStyle:{fontSize:20},showSeriesName:false}});
   applyPresentationChartFont(chart,{fontFamily:family});chartOwners.push(j+1);
   if(c.type==='line'){
     text(s,'2025-01-02',115,532,150,27,18,gray);
     text(s,'2025年末',655,532,150,27,18,gray);
     text(s,'2026-09-18',1060,532,155,27,18,gray);
   }
   const summary=d.items.join('；');
   if(summary)text(s,summary,64,c.horizontal?620:566,1150,c.horizontal?56:75,c.horizontal?19:22,gray);
   if(d.footnote)text(s,d.footnote,64,644,1150,25,16,gray);
 }else if(d.table){
   const vals=d.table;const h=vals.length*58;
   const t=s.tables.add({rows:vals.length,columns:2,left:68,top:219,width:1144,height:h,values:vals,columnWidths:[355,789]});
   t.borders.assign({fill:'#DCE5EB',width:1});
   for(let rr=0;rr<vals.length;rr++)for(let cc=0;cc<2;cc++){
      const cell=t.getCell(rr,cc);cell.fill=rr===0?navy:(rr%2?'#F2F6F9':'#FFFFFF');
      cell.text.style={typeface:family,fontSize:rr===0?27:25,color:rr===0?'#FFFFFF':navy,bold:rr===0,autoFit:'none'};
   }
   tableOwners.push(j+1);text(s,d.items.join('\n'),69,Math.min(230+h+10,562),1142,136,22,gray);
 }else if(d.kind==='flow'){
   d.items.forEach((v,k)=>{const x=56+k*302;box(s,x,278,268,200,k%2?'#EDF5F4':'#EDF1F6');text(s,String(k+1).padStart(2,'0'),x+20,297,220,42,31,teal,true);text(s,v,x+20,355,230,110,26);if(k<3)text(s,'›',x+275,326,26,60,45,gray);});
   text(s,'同一核心服务命令行、交互页面与提交报告',70,518,1130,70,32,navy,true);
   text(s,'每个实验记录数据、配置与源码身份，失败结果保留',70,591,1130,45,25,gray);
 }else if(d.kind==='timeline'){
   const stages=d.title.startsWith('年度')?[['此前三年','标签实现后训练'],['2023–2024','验证并保留选择'],['2025–2026/9','开发后历史复核']]:[['2020–2022','训练模型'],['2023–2024','验证并选择'],['2025–2026/9','冻结选择后评估']];
   stages.forEach(([a,b],k)=>{box(s,66+k*391,232,363,106,k===1?teal:navy);text(s,a,88+k*391,248,325,38,30,'#FFFFFF',true);text(s,b,88+k*391,289,325,34,24,'#FFFFFF');});
   items(s,d.items,381,26,61);
 }else if(d.formula_plain){
   box(s,67,228,1146,92,'#F3F6F9');text(s,d.formula_plain,85,254,1110,50,29,teal,true);items(s,d.items,357,26,69);
 }else{items(s,d.items,235,d.kind==='demo'?30:29,d.kind==='demo'?86:85);}
 s.speakerNotes.textFrame.setText(`建议用时：${d.seconds}秒\n${d.notes}\n${d.source??'来源：本项目 evidence/research_v2；完整定义与限制见最终报告。'}`);
}
await fs.mkdir(build,{recursive:true});
const candidate=path.join(build,'candidate.pptx');await (await PresentationFile.exportPptx(p)).save(candidate);
await fs.writeFile(path.join(build,'deck.proto.json'),JSON.stringify(p.toProto()));
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,pythonExecutable:python,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 explicitTotalSlideCount:20,requiredNativeChartOwnerSlides:chartOwners,requiredNativeTableOwnerSlides:tableOwners,
 materializeLiteralChartWorkbooks:true,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
 fontPolicy:{basis:'design',families:[family]},verifyArtifactToolImport:true,
 receiptPath:path.join(build,'validation-'+path.basename(final)+'.json')});
console.log(JSON.stringify({final,chartOwners,tableOwners,validated:true},null,2));
// Re-import the final file and render every final slide for visual QA.
const {FileBlob}=await import('@oai/artifact-tool');
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(final));
const renders=path.join(build,'render-'+path.basename(final,'.pptx'));await fs.mkdir(renders,{recursive:true});
for(let i=0;i<20;i++){
 const slide=finalDeck.slides.items[i];
 const blob=await finalDeck.export({slide,format:'png',scale:1});
 await fs.writeFile(path.join(renders,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
 console.log('Rendered '+(i+1));
}
