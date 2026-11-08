"""Reconstruct C-CCM configuration masses and a mass-only class diagnostic.

Python standard library only. Inputs are archived official metadata, not images,
checkpoints or predicted boxes. This program does not compute detection AP.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,json,math,zipfile

STATES=[(0,0),(1,1),(1,2),(2,1),(2,2),(3,1),(3,2)]
NAMES=['empty','pasta_50','pasta_90','rice_50','rice_90','water_50','water_90']

def load_bytes(blob):return json.loads(blob.decode('utf-8-sig'))
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run(metadata_zip,output_dir):
    names=['c_ccm_annotations.json','toolkit/annotations/ccm_train_annotation.json']
    metadata_zip=Path(metadata_zip).resolve();output_dir=Path(output_dir).resolve()
    package_root=Path(__file__).resolve().parents[1]
    if output_dir==package_root or package_root in output_dir.parents:
        raise ValueError('Outputs must be outside the code package')
    if metadata_zip.is_dir() and (output_dir==metadata_zip or metadata_zip in output_dir.parents):
        raise ValueError('Use an output directory separate from source metadata')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Use a fresh diagnostic output directory')
    if metadata_zip.is_dir():
        blobs={name:(metadata_zip/name).read_bytes() for name in names}
    else:
        with zipfile.ZipFile(metadata_zip) as archive:blobs={name:archive.read(name) for name in names}
    expected=json.loads((package_root/'configs/cccm/source_metadata_manifest.json').read_text(encoding='utf-8-sig'))['files']
    for name,blob in blobs.items():
        if hashlib.sha256(blob).hexdigest()!=expected[name]['sha256']:
            raise ValueError('Official metadata checksum mismatch: '+name)
    output_dir.mkdir(parents=True,exist_ok=True)
    native=load_bytes(blobs[names[0]]);ccm=load_bytes(blobs[names[1]])
    containers={int(c['id']):c for c in native['containers']}
    source_containers={c['name']:int(c['id']) for c in ccm['containers']}
    table=[]
    for container_id in range(1,7):
        name=containers[container_id]['name'];source_id=source_containers[name]
        for state,(filling_type,filling_level) in enumerate(STATES):
            rows=[a for a in ccm['annotations'] if (a['container id'],a['filling type'],a['filling level'])==(source_id,filling_type,filling_level)]
            triples={(float(a['container mass']),float(a['filling mass']),float(a['object mass'])) for a in rows}
            if len(rows)!=12 or len(triples)!=1:raise ValueError('Expected12annotations with one reference mass per configuration')
            container_mass,filling_mass,total_mass=next(iter(triples))
            if not math.isclose(container_mass+filling_mass,total_mass,abs_tol=1e-8):raise ValueError('Mass sum differs')
            table.append(dict(class_id=(container_id-1)*7+state,container_id=container_id,container_name=name,state=NAMES[state],
                filling_type=filling_type,filling_level=filling_level,container_mass_g=container_mass,
                filling_mass_g=filling_mass,total_mass_g=total_mass,source_rows=len(rows),
                source_annotation_ids=sorted(int(a['id']) for a in rows)))
    mass_by_class={r['class_id']:r['total_mass_g'] for r in table}
    frames=[]
    for row in native['annotations']:
        container_id=int(row['containerID'])
        if not 1<=container_id<=6:continue
        state=STATES.index((row['filling_type'],row['filling_level']))
        label=(container_id-1)*7+state
        frames.append(dict(image_id=int(row['id']),view=int(row['view']),class_id=label,mass_g=mass_by_class[label]))
    counts=dict(Counter(r['view'] for r in frames))
    if counts!={1:2929,2:2829,3:2969}:raise ValueError('Fixed view membership differs')
    train=[r for r in frames if r['view']==1];test=[r for r in frames if r['view']==3]
    means={label:sum(r['mass_g'] for r in train if r['class_id']==label)/sum(r['class_id']==label for r in train) for label in mass_by_class}
    correct=0;ties=0;per_class=defaultdict(lambda:dict(total=0,correct=0))
    for row in test:
        distances={label:abs(mean-row['mass_g']) for label,mean in means.items()};minimum=min(distances.values())
        candidates=sorted(label for label,distance in distances.items() if abs(distance-minimum)<1e-8)
        predicted=candidates[0];correct+=predicted==row['class_id'];ties+=len(candidates)>1
        per_class[row['class_id']]['total']+=1;per_class[row['class_id']]['correct']+=int(predicted==row['class_id'])
    groups=defaultdict(list)
    for label,mass in mass_by_class.items():groups[mass].append(label)
    result=dict(classes=len(table),distinct_reference_masses=len(groups),split_counts_by_view=counts,
        test_frames=len(test),correct=correct,classification_accuracy_pct=100*correct/len(test),tied_test_frames=ties,
        training_class_mass_means_g=means,per_class=dict(per_class),
        mass_collision_groups=[dict(total_mass_g=mass,classes=labels) for mass,labels in groups.items() if len(labels)>1],
        source_sha256={name:hashlib.sha256(blob).hexdigest() for name,blob in blobs.items()},
        input_construction='True container identity/content/filling annotations select the published configuration reference mass; same configuration shares one value.',
        rule='Nearest training-class mass mean; distances within1e-8g tie and smallest class index wins.',
        scope='Class-information diagnostic only. No images, neural network, boxes or COCO AP; not independent per-frame sensor validation.')
    dump(output_dir/'CONFIGURATION_MASS_TABLE.json',table)
    with (output_dir/'CONFIGURATION_MASS_TABLE.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        fields=[name for name in table[0] if name!='source_annotation_ids'];writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in table)
    dump(output_dir/'MASS_ONLY_CLASSIFICATION.json',result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--metadata-zip',type=Path)
    inputs.add_argument('--metadata-root',type=Path,help='Imported public_metadata directory')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();result=run(args.metadata_root or args.metadata_zip,args.output_dir)
    print(json.dumps({k:result[k] for k in ['classes','distinct_reference_masses','test_frames','correct','classification_accuracy_pct','tied_test_frames']},ensure_ascii=False))

if __name__=='__main__':main()
