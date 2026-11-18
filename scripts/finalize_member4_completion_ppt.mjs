import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
const workspaceDir=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=path.resolve(path.dirname(process.execPath),'../..');
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
process.env.RUNTIME_PYTHON=path.join(runtime,'python/python.exe');
const skill='C:/Users/Polaris/.codex/plugins/cache/openai-primary-runtime/presentations/26.1116.12010/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const build=path.join(workspaceDir,'tmp/member4_complete_ppt_20261118');
const result=await finalizePresentation({workspaceDir,candidatePath:path.join(build,'candidate.pptx'),
 finalPath:path.join(workspaceDir,'deliverables/member4_complete_20261118/MEMBER4_Presentation_4slides.pptx'),
 pythonExecutable:process.env.RUNTIME_PYTHON,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
             '--require-native-table-slide','2','--require-native-table-slide','4'],
 explicitTotalSlideCount:4,requiredNativeTableOwnerSlides:[2,4],
 fontPolicy:{basis:'design',families:['Arial'],scriptFonts:{ea:'Microsoft YaHei'}},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'final_validation.json')});
console.log(JSON.stringify(result));
