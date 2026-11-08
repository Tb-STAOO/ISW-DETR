"""Download and audit all predeclared C-CCM container 1..6 candidates.

Source and old pilot are read-only. New files are confined to OUT and DATA.
No training, outcome filtering, fabricated weighing, or silent split changes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import threading
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw
import requests

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
BASE = RUN.parent
SOURCE = BASE / 'CROSS_DOMAIN_DATASET_SEARCH_20261024_R1/robotics_containers'
DATA = Path('E:/ISW_DETR_external_data/CCCM_full_20261024')
OLD = Path('E:/ISW_DETR_external_data/CCM/images')
STATE = [(0, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
STATE_NAMES = ['empty', 'pasta_50', 'pasta_90', 'rice_50', 'rice_90', 'water_50', 'water_90']
SPLITS = {1: 'train', 2: 'dev', 3: 'test'}
MAX_RANGE = 32 * 1024 * 1024
MAX_GAP = 2 * 1024 * 1024
LOG_LOCK = threading.Lock()
HTTP_CONDITION = threading.Condition()
HTTP_ACTIVE = 0
HTTP_REDUCED_UNTIL = 0.
MAX_WORKERS = 4


class ServerBackoff(Exception):
    def __init__(self, seconds, status):
        super().__init__(f'HTTP {status} Retry-After {seconds}s')
        self.seconds, self.status = seconds, status


@contextmanager
def http_slot():
    global HTTP_ACTIVE
    with HTTP_CONDITION:
        while HTTP_ACTIVE >= (2 if time.monotonic() < HTTP_REDUCED_UNTIL else MAX_WORKERS):
            HTTP_CONDITION.wait(timeout=1.)
        HTTP_ACTIVE += 1
    try:
        yield
    finally:
        with HTTP_CONDITION:
            HTTP_ACTIVE -= 1
            HTTP_CONDITION.notify_all()


def reduce_workers(seconds):
    global HTTP_REDUCED_UNTIL
    with HTTP_CONDITION:
        HTTP_REDUCED_UNTIL = max(HTTP_REDUCED_UNTIL, time.monotonic() + seconds + 300.)
        HTTP_CONDITION.notify_all()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(value):
    with LOG_LOCK:
        with (OUT / 'TRANSPORT_EVENTS.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(utc=utc(), **value), ensure_ascii=False) + '\n')


def photo_path(row):
    return DATA / 'images' / f'{row["image_id"]:06d}.png'


def dhash(image):
    arr = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
    return f'{int.from_bytes(np.packbits(arr[:, 1:] > arr[:, :-1]).tobytes(), "big"):016x}'


def validate_png(row, blob, origin):
    entry = row['source']
    if len(blob) != entry['size'] or (zlib.crc32(blob) & 0xffffffff) != entry['crc32']:
        raise ValueError(f'PNG member size/CRC mismatch {entry["name"]}')
    with Image.open(BytesIO(blob)) as im:
        im.verify()
    with Image.open(BytesIO(blob)) as im:
        im.load()
        if im.mode != 'RGB' or im.size != (1280, 720):
            raise ValueError(f'Unexpected native image {entry["name"]}: {im.mode} {im.size}')
        pixel_hash = hashlib.sha256(im.tobytes()).hexdigest()
        crop = im.crop(tuple(row['native_bbox_xyxy']))
        frame_dhash, crop_dhash = dhash(im), dhash(crop)
    return dict(image_id=row['image_id'], split=row['split'], category_id=row['category_id'],
        relative_path=photo_path(row).relative_to(DATA).as_posix(), origin=origin,
        archive=entry['archive'], source_member=entry['name'], bytes=len(blob),
        crc32=entry['crc32'], sha256=hashlib.sha256(blob).hexdigest(), pixel_sha256=pixel_hash,
        frame_dhash=frame_dhash, native_box_crop_dhash=crop_dhash, mode='RGB',
        width=1280, height=720, member_size_crc_valid=True, decoded=True)


def persist_member(row, blob, origin):
    result = validate_png(row, blob, origin)
    path = photo_path(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix('.png.part')
        tmp.write_bytes(blob)
        tmp.replace(path)
    return result


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    native = read(SOURCE / 'c_ccm_annotations.json')
    members = {m['name']: m for m in read(SOURCE / 'CCCM_zip_members.json')}
    masses = read(SOURCE / 'toolkit/annotations/ccm_train_annotation.json')
    containers = {c['id']: c for c in native['containers']}
    source_containers = {c['name']: c['id'] for c in masses['containers']}
    images = {i['image_id']: i for i in native['images']}
    table, categories = {}, []
    for cid in range(1, 7):
        name = containers[cid]['name']
        source_id = source_containers[name]
        for state, (fill, level) in enumerate(STATE):
            source = [a for a in masses['annotations'] if (a['container id'], a['filling type'], a['filling level']) == (source_id, fill, level)]
            values = {(float(a['container mass']), float(a['filling mass']), float(a['object mass'])) for a in source}
            assert len(source) == 12 and len(values) == 1
            cm, fm, tm = next(iter(values))
            assert cm > 0 and fm >= 0 and tm > 0 and math.isclose(cm + fm, tm, abs_tol=1e-8)
            leaf = (cid - 1) * 7 + state
            category = dict(id=leaf, name=f'{name}/{STATE_NAMES[state]}', identity_id=cid-1, state_id=state,
                native_container_id=cid, native_filling_type=fill, native_filling_level=level)
            categories.append(category)
            table[leaf] = dict(category_id=leaf, container_name=name, identity_id=cid-1, state_id=state,
                native_container_id=cid, native_mass_container_id=source_id, native_filling_type=fill,
                native_filling_level=level, container_mass_g=cm, filling_mass_g=fm, object_mass_g=tm,
                source_annotation_ids=[a['id'] for a in source])
    candidates = []
    for a in sorted(native['annotations'], key=lambda a: a['id']):
        cid = int(a['containerID'])
        if cid not in range(1, 7):
            continue
        iid = int(a['id'])
        im = images[iid]
        assert im['filename'].strip() == f'{iid:06d}.png'
        assert im['width'] == 1280 and im['height'] == 720 and im['color'] == 'RGB'
        state = STATE.index((a['filling_type'], a['filling_level']))
        box = list(map(float, a['bbox']))
        assert len(box) == 4 and all(math.isfinite(x) for x in box)
        x1, y1, x2, y2 = box
        assert 0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 720
        candidates.append(dict(image_id=iid, split=SPLITS[a['view']], view=a['view'],
            category_id=(cid-1)*7+state, identity_id=cid-1, state_id=state,
            native_container_id=cid, native_filling_type=a['filling_type'], native_filling_level=a['filling_level'],
            native_occlusion=a['occlusion'], native_transparency=a['transparency'], native_scenario=a['scenario'],
            native_bbox_xyxy=box, source=members[f'rgb/{iid:06d}.png']))
    assert len(candidates) == 8727 and len({r['image_id'] for r in candidates}) == 8727
    counts = dict(Counter(r['split'] for r in candidates))
    assert counts == {'train': 2929, 'dev': 2829, 'test': 2969}
    source_files = ['c_ccm_annotations.json', 'CCCM_zip_members.json', 'toolkit/annotations/ccm_train_annotation.json']
    protocol = dict(created_utc=utc(), status='source inventory and view split locked before full-data training',
        dataset='C-CCM all native matching containers 1..6', data_root=str(DATA), num_classes=42,
        num_identities=6, num_states=7, categories=categories, leaf_to_parent=[i//7 for i in range(42)],
        split_views=dict(train=1, dev=2, test=3), counts=counts,
        candidate_rule='All native container IDs 1..6; no performance-based selection or difficulty exclusions',
        source_annotations=str(SOURCE/'c_ccm_annotations.json'),
        mass_source_annotations=str(SOURCE/'toolkit/annotations/ccm_train_annotation.json'),
        source_hashes={name:digest(SOURCE/name) for name in source_files},
        source_urls=['https://zenodo.org/records/4642577', 'https://corsmal.github.io/filling.html'],
        image_license=native['licenses'], mass_annotation_license=masses['licenses'],
        excluded_native_container_ids=[7,8],
        exclusion_reason='Different physical container names than CCM mass containers 7/8; not missing-value imputation',
        image_input='Uncropped native RGB; annotations retained as targets only; crop descriptors are audit-only',
        mass_definition='Author configuration-level total mass in grams, exact container name + native filling type/level join; no independent per-frame weighing',
        telemetry_available=False, one_target_one_static_record=True,
        observations_schema='Only image_id, mass_unit, records record_id/mass_g, telemetry_available; no class/owner fields',
        owners_separate=True, prior_source='training observations and training owner mapping only; std floor 1g',
        previous_pilot_test_observed=True, blind_test_claim=False,
        known_physical_containers_shared_across_views=True, source_recording_ids_available=False,
        duplicate_policy='Report cross-split exact compressed/pixel duplicate groups; stop ready status for root decision, never silently reassign or delete',
        source_candidate_file=str(OUT/'FULL_CANDIDATE_LIST.json'), mass_lookup_table=str(OUT/'CONFIGURATION_MASS_TABLE.json'),
        selection_file=str(OUT/'FIXED_SELECTION.json'),
        selection_sha256=hashlib.sha256(json.dumps([dict(image_id=r['image_id'],split=r['split'],category_id=r['category_id'])
            for r in candidates],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        source_conversion_file=str(Path(__file__).resolve()),source_conversion_sha256=digest(__file__),
        mass_source_repeats_per_configuration=12,mass_source_rows=684,num_parents=6,
        transport=dict(max_workers=MAX_WORKERS, on_429_or_503_workers=2, reduced_worker_cooldown_seconds=300,
            endpoint='Same official Zenodo record API /files/{archive}/content; bounded Range verified',
            range_bytes_cap=MAX_RANGE, max_gap_bytes=MAX_GAP,
            whole_archives_downloaded=False, member_crc_sha_pixel_decode_required=True),
        all_images_downloaded=False)
    old_path = OUT/'SOURCE_PROTOCOL.json'
    if old_path.exists():
        old = read(old_path)
        for key in ['counts','categories','source_hashes','split_views','data_root','candidate_rule']:
            assert old[key] == protocol[key], f'Locked protocol differs: {key}'
        if old.get('source_conversion_sha256') != protocol['source_conversion_sha256'] or old.get('transport') != protocol['transport']:
            old.setdefault('transport_revisions',[]).append(dict(utc=utc(),
                reason='Root-authorized official endpoint and bounded parallel transport optimization before training',
                old_transport=old.get('transport'),new_transport=protocol['transport'],
                old_preparation_sha256=old.get('source_conversion_sha256'),
                new_preparation_sha256=protocol['source_conversion_sha256'],
                candidate_split_mass_and_model_hyperparameters_unchanged=True))
            old['transport']=protocol['transport']
            old['source_conversion_sha256']=protocol['source_conversion_sha256']
            write(old_path,old)
    else:
        write(old_path, protocol)
    write(OUT/'FULL_CANDIDATE_LIST.json', candidates)
    write(OUT/'FIXED_SELECTION.json',[dict(image_id=r['image_id'],split=r['split'],category_id=r['category_id']) for r in candidates])
    write(OUT/'CONFIGURATION_MASS_TABLE.json', list(table.values()))
    train_counts=Counter(r['category_id'] for r in candidates if r['split']=='train')
    write(DATA/'taxonomy.json', dict(categories=categories, leaf_to_parent=[i//7 for i in range(42)],
        leaf_to_state=[i%7 for i in range(42)], num_classes=42,num_parents=6,num_identities=6,num_states=7,
        leaf_counts=[train_counts[i] for i in range(42)],leaf_counts_source='train only'))
    emit_dataset_files(candidates, table, categories)
    return candidates, table, categories


def emit_dataset_files(candidates, table, categories):
    stats = {}
    priors = []
    for view, split in SPLITS.items():
        photos, annotations, observations, owners = [], [], [], []
        rows = [r for r in candidates if r['split'] == split]
        for r in rows:
            iid = r['image_id']; leaf = r['category_id']; x1,y1,x2,y2 = r['native_bbox_xyxy']
            box = [x1,y1,x2-x1,y2-y1]; mass = table[leaf]['object_mass_g']; rid = f'{iid:06d}_0'
            photos.append(dict(id=iid, file_name=f'images/{iid:06d}.png', width=1280,height=720,
                native_image_id=iid,native_view=view,split=split))
            annotations.append(dict(id=iid,image_id=iid,category_id=leaf,bbox=box,area=box[2]*box[3],
                iscrowd=0,native_annotation_id=iid))
            observations.append(dict(image_id=iid,mass_unit='g',records=[dict(record_id=rid,mass_g=mass)],telemetry_available=False))
            owner=dict(record_id=rid,annotation_id=iid,category_id=leaf,identity_id=r['identity_id'],
                state_id=r['state_id'],bbox=box,mass_g=mass,native_container_id=r['native_container_id'],
                native_filling_type=r['native_filling_type'],native_filling_level=r['native_filling_level'])
            owners.append(dict(image_id=iid,split=split,view=view,owners=[owner]))
        write(DATA/f'annotations/{split}.json',dict(info=dict(description='All matching C-CCM native configurations; cross-view split'),
            images=photos,annotations=annotations,categories=categories))
        write(DATA/f'observations/{split}.json',observations)
        write(DATA/f'evaluation_sidecars/{split}_owners.json',owners)
        counts = Counter(r['category_id'] for r in rows)
        assert len(counts) == 42
        stats[split] = dict(images=len(rows),annotations=len(rows),view=view,
            class_counts={str(k):counts[k] for k in range(42)},occlusion_counts=dict(Counter(r['native_occlusion'] for r in rows)))
        if split == 'train':
            for leaf in range(42):
                values = [p['owners'][0]['mass_g'] for p in owners if p['owners'][0]['category_id'] == leaf]
                priors.append(dict(category_id=leaf,count=len(values),mean_g=float(np.mean(values)),
                    empirical_std_g=float(np.std(values,ddof=1)),std_g=max(1.,float(np.std(values,ddof=1)))))
    write(DATA/'priors/train_mass_priors.json',dict(source='train owners + observations only',
        owners_sha256=digest(DATA/'evaluation_sidecars/train_owners.json'),
        observations_sha256=digest(DATA/'observations/train.json'),std_floor_g=1.,classes=priors))
    write(OUT/'CLASS_SPLIT_COUNTS.json',stats)


def range_batches(rows):
    archives = defaultdict(list)
    for row in rows:
        archives[row['source']['archive']].append(row)
    batches = []
    for name in sorted(archives):
        current = []; start = end = None
        for row in sorted(archives[name],key=lambda r:r['source']['offset']):
            ent = row['source']; a = ent['offset']
            b = min(ent['archive_bytes']-1,a+30+len(ent['name'].encode())+1024+ent['compressed_size']-1)
            if current and (b-start+1 > MAX_RANGE or a-end > MAX_GAP):
                batches.append((start,end,current)); current=[]
            if not current: start=a;end=b
            else: end=max(end,b)
            current.append(row)
        if current: batches.append((start,end,current))
    return batches


def retry_delay(value, fallback):
    if not value: return fallback
    try: return max(0.,float(value))
    except ValueError:
        try:return max(0.,(parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds())
        except Exception:return fallback


def fetch_batch(batch):
    start,end,rows=batch; cap=end-start+1
    assert 0 < cap <= MAX_RANGE
    url=f'https://zenodo.org/api/records/4642577/files/{rows[0]["source"]["archive"]}/content'
    for attempt in range(9):
        try:
            tick=time.monotonic()
            with http_slot():
                with requests.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},
                        timeout=(20,120),stream=True) as response:
                    if response.status_code in (429,503):
                        delay=retry_delay(response.headers.get('Retry-After'),min(180.,15.*(attempt+1)))
                        reduce_workers(delay)
                        raise ServerBackoff(delay,response.status_code)
                    response.raise_for_status()
                    if response.status_code!=206:raise ValueError('Server ignored bounded HTTP Range; refused download')
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if not match or tuple(map(int,match.groups()))!=(start,end,rows[0]['source']['archive_bytes']):
                        raise ValueError(f'Unexpected bounded Content-Range {response.headers.get("Content-Range")}')
                    data=bytearray()
                    for chunk in response.iter_content(1024*1024):
                        data.extend(chunk)
                        if len(data)>cap:raise ValueError('Response exceeded locked Range size')
                    if len(data)!=cap:raise ValueError('Truncated bounded Range')
            results=[]
            for row in rows:
                ent=row['source'];offset=ent['offset']-start
                header=struct.unpack('<IHHHHHIIIHH',data[offset:offset+30])
                if header[0]!=0x04034b50:raise ValueError('Bad ZIP local header')
                compressed_at=offset+30+header[-2]+header[-1]
                compressed=data[compressed_at:compressed_at+ent['compressed_size']]
                if header[3]==8:blob=zlib.decompress(compressed,-15)
                elif header[3]==0:blob=bytes(compressed)
                else:raise ValueError(f'Unsupported ZIP compression {header[3]}')
                results.append(persist_member(row,blob,'downloaded_bounded_range'))
            event(dict(kind='range_complete',archive=rows[0]['source']['archive'],start=start,end=end,
                downloaded_bytes=cap,members=len(rows),elapsed_seconds=time.monotonic()-tick))
            return results,cap
        except ServerBackoff as exc:
            event(dict(kind='server_backoff',status=exc.status,seconds=exc.seconds,
                max_workers_temporarily=2,attempt=attempt+1))
            print(f'HTTP {exc.status}; Retry-After {exc.seconds}s; temporarily using two HTTP workers',flush=True)
            time.sleep(exc.seconds)
        except Exception as exc:
            event(dict(kind='range_retry',archive=rows[0]['source']['archive'],start=start,end=end,
                attempt=attempt+1,error=repr(exc)))
            print('Range retry',attempt+1,repr(exc),flush=True)
            if attempt==8:raise
            time.sleep(min(120.,5.*(attempt+1)))
    raise RuntimeError('Bounded Range retries exhausted')


def download(candidates):
    verified=[];pending=[];started=time.monotonic();network_bytes=0
    def progress(stage,**extra):
        write(OUT/'DATA_PREPARATION_PROGRESS.json',dict(utc=utc(),stage=stage,expected=len(candidates),
            verified=len(verified),pending=len(candidates)-len(verified),data_root=str(DATA),
            network_bytes_this_execution=network_bytes,elapsed_seconds=time.monotonic()-started,**extra))
    progress('verifying_existing_and_old_pilot')
    for idx,row in enumerate(candidates):
        path=photo_path(row);old=OLD/path.name
        existing=path if path.exists() else old if old.exists() else None
        if existing:
            try:
                blob=existing.read_bytes()
                record=validate_png(row,blob,'new_root_resume' if existing==path else 'verified_old_pilot_copy')
                if existing!=path:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(existing,path)
                verified.append(record)
            except Exception as exc:
                event(dict(kind='existing_image_failed',image_id=row['image_id'],path=str(existing),error=repr(exc)))
                # A corrupt existing new-root file is not silently accepted.
                if existing==path:
                    quarantine=DATA/'quarantine'/f'{row["image_id"]:06d}_{int(time.time())}.png'
                    quarantine.parent.mkdir(parents=True,exist_ok=True);path.replace(quarantine)
                pending.append(row)
        else:pending.append(row)
        if idx%200==0:progress('verifying_existing_and_old_pilot',scanned=idx+1)
    batches=range_batches(pending)
    write(OUT/'RANGE_PLAN.json',[dict(archive=rs[0]['source']['archive'],start=a,end=b,bytes=b-a+1,
        image_ids=[r['image_id'] for r in rs]) for a,b,rs in batches])
    progress('downloading',range_batches=len(batches),completed_range_batches=0)
    print(f'Verified cached {len(verified)}/{len(candidates)}, pending {len(pending)}, bounded ranges {len(batches)}',flush=True)
    with (OUT/'IMAGE_VERIFICATION_JOURNAL.jsonl').open('a',encoding='utf-8') as journal:
        for record in verified:journal.write(json.dumps(record)+'\n')
        journal.flush()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            jobs={pool.submit(fetch_batch,b):b for b in batches}
            for finished,job in enumerate(as_completed(jobs),1):
                try:records,downloaded=job.result()
                except Exception as exc:
                    progress('transport_failed_resumable',error=repr(exc))
                    raise
                verified.extend(records);network_bytes+=downloaded
                for record in records:journal.write(json.dumps(record)+'\n')
                journal.flush()
                progress('downloading',range_batches=len(batches),completed_range_batches=finished)
                print(f'Verified PNG {len(verified)}/{len(candidates)}; Range {finished}/{len(batches)}; net {network_bytes/2**20:.1f} MiB',flush=True)
    assert len(verified)==len(candidates)
    verified.sort(key=lambda r:r['image_id'])
    write(OUT/'IMAGE_MANIFEST.json',dict(scope='All selected PNG members; per-member size CRC SHA256 RGB decode pixel hash, not whole ZIP checksum',
        created_utc=utc(),expected=len(candidates),verified=len(verified),members=verified))
    progress('deduplication_diagnostics')
    return verified


def duplicate_diagnostics(candidates,verified):
    results={}
    for field in ['sha256','pixel_sha256']:
        groups=defaultdict(list)
        for r in verified:groups[r[field]].append(dict(image_id=r['image_id'],split=r['split'],category_id=r['category_id']))
        repeated=[dict(hash=k,members=v,cross_split=len({r['split'] for r in v})>1) for k,v in groups.items() if len(v)>1]
        results[field]=dict(duplicate_groups=repeated,total_groups=len(repeated),
            cross_split_groups=sum(g['cross_split'] for g in repeated),
            cross_split_images=sum(len(g['members']) for g in repeated if g['cross_split']))
    # Crop dHash similarity is a diagnostic, not evidence of duplicated acquisition.
    # Compare only same native configuration across distinct views; retain every image.
    byleaf=defaultdict(list)
    for r in verified:byleaf[r['category_id']].append(r)
    summary=[];examples=[]
    for leaf,rows in sorted(byleaf.items()):
        pairs=0;near=0;minimum=65
        for i,a in enumerate(rows):
            av=int(a['native_box_crop_dhash'],16)
            for b in rows[i+1:]:
                if a['split']==b['split']:continue
                distance=(av^int(b['native_box_crop_dhash'],16)).bit_count();pairs+=1
                minimum=min(minimum,distance)
                if distance<=4:
                    near+=1
                    if len(examples)<200:examples.append(dict(category_id=leaf,image_a=a['image_id'],image_b=b['image_id'],
                        split_a=a['split'],split_b=b['split'],crop_dhash_hamming=distance))
        summary.append(dict(category_id=leaf,cross_view_pairs_compared=pairs,
            crop_dhash_hamming_le4_pairs=near,minimum_cross_view_crop_dhash_hamming=minimum))
    results['near_duplicate_diagnostic']=dict(method='Audit-only native-box crop 64-bit difference hash; cross-view pairs within same native class',
        limitations='Small visually similar crops can hash alike; not a recording/frame-identity test or automatic deletion rule',
        no_images_removed=True,class_statistics=summary,first_200_pairs=examples)
    write(OUT/'DUPLICATE_DIAGNOSTICS.json',results)
    return results


def preview(candidates):
    selected=[]
    for identity in range(6):
        canvas=Image.new('RGB',(3*426,7*265),'white');draw=ImageDraw.Draw(canvas)
        for state in range(7):
            for view in (1,2,3):
                rows=[r for r in candidates if r['identity_id']==identity and r['state_id']==state and r['view']==view]
                # Prefer occluded frames if available to ensure the audit includes hard cases.
                row=min(rows,key=lambda r:(r['native_occlusion']!=1,
                    hashlib.sha256(f'cccm-full-box-review-v1|{r["image_id"]}'.encode()).hexdigest()))
                with Image.open(photo_path(row)) as im:
                    im=im.copy();ImageDraw.Draw(im).rectangle(row['native_bbox_xyxy'],outline='red',width=5)
                    im.thumbnail((426,240));x=(view-1)*426;y=state*265
                    canvas.paste(im,(x,y+25));draw.text((x+3,y+4),f'{STATE_NAMES[state]} v{view} #{row["image_id"]} occ={row["native_occlusion"]}',fill='black')
                selected.append({k:row[k] for k in ['image_id','identity_id','state_id','view','split','native_bbox_xyxy','native_occlusion']})
        canvas.save(OUT/f'BOX_REVIEW_IDENTITY_{identity}.png')
    write(OUT/'BOX_REVIEW_SELECTION.json',dict(rule='Each identity x state x view cell, prefer native occlusion=1, then smallest SHA256(cccm-full-box-review-v1|image_id)',
        sample_count=len(selected),samples=selected,human_visual_review_done=False))


def finalize(candidates,verified,duplicates):
    cross=sum(duplicates[field]['cross_split_groups'] for field in ['sha256','pixel_sha256'])
    output_paths=[DATA/f'{folder}/{split}{suffix}.json' for folder,suffix in [('annotations',''),('observations',''),('evaluation_sidecars','_owners')]
        for split in ['train','dev','test']]
    output_paths += [DATA/'taxonomy.json',DATA/'priors/train_mass_priors.json']
    ready=dict(created_utc=utc(),status='cross_split_exact_duplicate_requires_root_decision' if cross else 'ready_for_independent_audit',
        expected_images=8727,verified_images=len(verified),all_member_size_crc_sha256_passed=True,
        all_native_png_rgb_decode_passed=True,all_native_boxes_metadata_valid=True,
        exact_cross_split_duplicate_groups_counting_both_hash_types=cross,
        source_protocol_sha256=digest(OUT/'SOURCE_PROTOCOL.json'),
        preparation_script_sha256=digest(__file__),manifest_sha256=digest(OUT/'IMAGE_MANIFEST.json'),
        output_sha256={str(p):digest(p) for p in output_paths},
        split_counts=dict(Counter(r['split'] for r in candidates)),
        no_performance_filtering=True,no_training_done=True,previous_pilot_test_observed=True,
        independent_source_and_loader_audit_remaining=True,human_preview_review_remaining=True)
    write(OUT/'DATA_READY_QA.json',ready)
    write(OUT/'DATA_PREPARATION_PROGRESS.json',dict(utc=utc(),stage=ready['status'],expected=8727,
        verified=len(verified),pending=0,data_root=str(DATA)))
    text=f'''# C-CCM 全候选准备结果\n\n状态：{ready['status']}。共 8,727 张，每成员完成长度、原始 ZIP CRC32、SHA256、PIL 解码 RGB/1280×720 和解码像素 SHA256 核查。\n\n固定原生视角划分为 train=view1 2,929 张、dev=view2 2,829 张、test=view3 2,969 张，42 联合类别/6 身份/7 状态全部覆盖。源元数据和旧 pilot 均未修改，没有按性能或可见难度筛选图片。\n\n全图与像素哈希的跨划分重复组总计（同一组可能按两种哈希重复计算）：{cross}；具体见 DUPLICATE_DIAGNOSTICS.json。目标框裁剪感知哈希只用于诊断，不用于训练输入，也未据此删除图片。\n\n数据根：`{DATA}`。COCO 标签、仅含质量值的 observations、GT owners sidecar 分开保存；先验只从训练 owner/observation 配对拟合，1 g 为静态 SD 下限，不代表测量方差。全部质量为作者配置级总质量，不是逐图新称重；没有构造真实事务门控。\n\n当前 test 视角中旧 pilot 的 168 张结果已经被观察，不能称为新盲测。六张原图框预览覆盖 6×7×3=126 个单元，并优先包含遮挡；人工几何复查与独立审计仍单独执行。\n'''
    (OUT/'DATA_PREPARATION_CN.md').write_text(text,encoding='utf-8')
    print(json.dumps(ready,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--setup-only',action='store_true');args=parser.parse_args()
    candidates,table,categories=setup()
    print('Full source inventory and provisional view-based annotations emitted',len(candidates),flush=True)
    if args.setup_only:return
    verified=download(candidates)
    duplicates=duplicate_diagnostics(candidates,verified)
    preview(candidates)
    finalize(candidates,verified,duplicates)


if __name__=='__main__':main()
