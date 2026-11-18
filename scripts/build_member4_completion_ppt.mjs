// Create the updated four-slide contribution from verified experiment artifacts.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

// fileURLToPath is required for Unicode Windows workspace paths.
const {fileURLToPath}=await import('node:url');
const workspace=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const build=path.join(workspace,'tmp/member4_complete_ppt_20261118');
const output=path.join(workspace,'deliverables/member4_complete_20261118');
const runtime=path.resolve(path.dirname(process.execPath),'../..');
const modules=path.join(runtime,'node/node_modules');
process.env.RUNTIME_NODE_MODULES=modules;
process.env.RUNTIME_PYTHON=path.join(runtime,'python/python.exe');
const require=createRequire(path.join(modules,'__member4_final__.cjs'));
const {Presentation,PresentationFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
await fs.mkdir(build,{recursive:true});
const data=JSON.parse(await fs.readFile(path.join(output,'presentation_data.json'),'utf8'));
const gallery=JSON.parse(await fs.readFile(path.join(output,'gallery/index.json'),'utf8'));
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const color={ink:'#142D4E',body:'#304356',accent:'#087F8C',light:'#EFF4F7'};
const pos=(left,top,width,height)=>({left,top,width,height});
function text(slide,value,x,y,w,h,size=28,bold=false,fill=color.body){
 const s=slide.shapes.add({geometry:'textbox',position:pos(x,y,w,h),fill:'none',line:{fill:'none',width:0}});
 s.text=value;s.text.style={typeface:'Arial',fontSize:size,bold,color:fill};return s;
}
function page(title){const slide=deck.slides.add();slide.background.fill='#FFFFFF';text(slide,title,56,34,1168,70,46,true,color.ink);return slide;}
function table(slide,values,x,y,w,h,widths){
 const t=slide.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
 t.borders.assign({style:'solid',fill:'#D7E0E7',width:1});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?color.ink:r%2?'#FFFFFF':color.light;
  cell.text.style={typeface:'Arial',fontSize:26,bold:r===0,color:r===0?'#FFFFFF':color.body};
 }
 return t;
}
async function picture(slide,file,box,alt,crop){
 slide.images.add({blob:new Uint8Array(await fs.readFile(file)),contentType:/\.jpe?g$/i.test(file)?'image/jpeg':'image/png',
                  position:box,fit:crop?'cover':'contain',alt,...(crop?{crop}:{})});
}
const pct=v=>(v*100).toFixed(2);
const clean=(m,k)=>data.summary.ABCD[m][k].clean_percent.toFixed(2);
const signed=v=>(v>=0?'+':'')+v.toFixed(3);

const s1=page('Photometric augmentation');
text(s1,'50% probability      One operation      Strength: 0.8–1.2',56,121,1168,48,31,true,color.accent);
const labels=['Original','Brightness ×0.8','Contrast ×0.8','Gamma 0.8'];
for(let i=0;i<4;i++){
 const x=56+i*296;
 text(s1,labels[i],x,198,280,43,27,true,color.ink);
 const px=4+i*260,py=103,pw=252,ph=252;
 const crop={left:px/1040,top:py/660,right:(1040-px-pw)/1040,bottom:(660-py-ph)/660};
 await picture(s1,path.join(workspace,'runs/member4_stage2_20261115_v2/visuals/01.png'),pos(x,251,280,280),
               labels[i]+' on the same verified training image; example strength 0.8',crop);
}
text(s1,'Unchanged masks      After resize; before normalization',56,566,1168,44,30,true);
text(s1,'C = A + augmentation        D = B + augmentation',56,628,1168,42,30);

const s2=page('Four-model results on clean images');
text(s2,'4,428 images      Shared initialization      30 epochs',56,124,1168,43,30);
table(s2,[['Model','IoU (%)','F1 (%)','Boundary IoU (%)'],...['A','B','C','D'].map(m=>[
 m,clean(m,'foreground_iou'),clean(m,'region_f1'),clean(m,'boundary_iou')])],56,202,1168,310,[220,270,270,408]);
text(s2,'Small differences      Single training seed',56,553,1168,45,34,true,color.accent);
text(s2,'Validation-only selection      Higher scores = better overlap',56,624,1168,43,29);

const s3=page('Robustness to photometric shifts');
text(s3,'4,428 images per condition      Shared, zoomed IoU axis',56,114,1168,42,29);
await picture(s3,path.join(output,'figures/ABCD_robustness_presentation.png'),pos(30,164,1220,415),
 'Python-generated scientific plot: all four models, three perturbation families, complete test set at every point');
text(s3,`Mean IoU change: C − A ${signed(data.augmentation_effects_pp.C_minus_A)} pp    D − B ${signed(data.augmentation_effects_pp.D_minus_B)} pp`,56,590,1168,46,31,true,color.accent);
text(s3,'12 perturbations      Clean excluded from mean      Single seed',56,641,1168,40,27);

const s4=page('Transfer to GSD glass scenes');
text(s4,'813 frozen records      No GSD training or tuning',56,118,1168,43,30);
table(s4,[['Model','IoU (%)','Boundary IoU (%)','Background FPR (%)'],...['A','D'].map(m=>[
 m,pct(data.summary.GSD[m].foreground_iou),pct(data.summary.GSD[m].boundary_iou),pct(data.summary.GSD[m].background_fpr)])],
 56,180,1168,146,[150,260,350,408]);
const sample=gallery.find(r=>r.dataset==='GSD'&&r.id==='gsd/test/122355684_81bf2b3d28');
if(!sample)throw new Error('Declared GSD failure is absent from verified gallery');
const items=[['Original',sample.image],['Ground truth',sample.ground_truth],
 ['A prediction',sample.predictions.clean.A.mask],['D prediction',sample.predictions.clean.D.mask]];
for(let i=0;i<items.length;i++){
 const x=56+296*i;
 text(s4,items[i][0],x,347,280,37,26,true,color.ink);
 await picture(s4,path.join(output,'gallery',items[i][1]),pos(x,393,280,220),
               items[i][0]+' for the declared low-D-IoU open-window failure');
}
text(s4,'Fewer background errors      No overall IoU gain',56,644,1168,42,31,true,color.accent);

for(let i=0;i<4;i++){
 const n=data.slides[i];
 deck.slides.items[i].speakerNotes.text=`Suggested time: ${n.seconds} seconds\n\n中文理解\n${n.zh}\n\nEnglish\n${n.en}\n\nSources\n${n.source}\nGallery case selected by the declared rule; never used to alter test membership.`;
 const png=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
 const layout=await deck.slides.items[i].export({format:'layout'});
 await fs.writeFile(path.join(build,`slide-${i+1}.layout.json`),new Uint8Array(await layout.arrayBuffer()));
}
await(await PresentationFile.exportPptx(deck)).save(path.join(build,'candidate.pptx'));
console.log(JSON.stringify({candidate:path.join(build,'candidate.pptx'),previews:[1,2,3,4].map(i=>path.join(build,`slide-${i}.png`))}));
