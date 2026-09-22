from __future__ import annotations
import json
from pathlib import Path
import laspy
from .seam_reconcile import build_fastgc_seam_overrides, apply_overrides_to_las

def finalize_fastgc_seams_in_place(processed_root):
    processed_root=Path(processed_root); workspace=processed_root.parent; mf=workspace/'tile_manifest.json'
    report={'applied':False,'tiles_changed':0,'points_changed':0}
    if not mf.exists(): return report
    manifest=json.loads(mf.read_text(encoding='utf-8'))
    if bool(manifest.get('use_existing_tiles',False)): return report
    gc_root=processed_root/'FAST_GC'
    if not gc_root.exists(): return report
    report_path=workspace/'FAST_GC_prederive_seam_qc.json'; overrides=build_fastgc_seam_overrides(manifest,processed_root,product='FAST_GC',report_path=report_path)
    for tile_name,ov in overrides.items():
        if ov is None: continue
        fp=gc_root/tile_name
        if not fp.exists(): continue
        las=laspy.read(fp); before=las.classification.copy(); apply_overrides_to_las(las,ov); changed=int((before!=las.classification).sum())
        if changed: las.write(fp); report['tiles_changed']+=1; report['points_changed']+=changed
    report['applied']=True; report['report_path']=str(report_path); return report
