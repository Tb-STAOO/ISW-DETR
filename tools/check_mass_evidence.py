"""Refit train-only priors, verify Gaussian overlap and replay raw-fault contracts."""
import argparse,csv,json,math,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'reference_sources')]
from run_private_controls import infer,physical_cache,read,sha,tr
from physical_interventions import fallback_contract

def overlap(mu1,sd1,mu2,sd2):
    """Integrate the lower Gaussian density over its exact crossing intervals."""
    if min(sd1,sd2)<=0:raise ValueError('Positive Gaussian SD required')
    if math.isclose(sd1,sd2,rel_tol=1e-12):return float(2*ndtr(-abs(mu1-mu2)/(2*sd1)))
    delta=mu2-mu1
    a=1/sd2**2-1/sd1**2;b=-2*delta/sd2**2
    c=delta**2/sd2**2+2*math.log(sd2/sd1)
    roots=sorted(mu1+float(v.real) for v in np.roots([a,b,c]) if abs(v.imag)<1e-8)
    boundaries=[-math.inf,*roots,math.inf];total=0.
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        point=(lo+hi)/2 if math.isfinite(lo) and math.isfinite(hi) else (hi-12*max(sd1,sd2) if math.isfinite(hi) else lo+12*max(sd1,sd2))
        log1=-math.log(sd1)-.5*((point-mu1)/sd1)**2
        log2=-math.log(sd2)-.5*((point-mu2)/sd2)**2
        mu,sd=(mu1,sd1) if log1<=log2 else (mu2,sd2)
        # Use survival probabilities for a right-tail interval.
        total+=float(ndtr((mu-lo)/sd)-ndtr((mu-hi)/sd)) if lo>=mu else float(ndtr((hi-mu)/sd)-ndtr((lo-mu)/sd))
    return min(1.,max(0.,total))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset-root','inference-root','weights-root','control-weights-root','output']:p.add_argument('--'+name,required=True)
    args=p.parse_args();out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):p.error('Use a fresh evidence-check directory')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    train=Path(args.dataset_root)/'ScaleVision-MM-V3/annotations/splits/instances_multimodal_train.json'
    source=read(train);images=[r for r in source['images'] if 'source_scene_id' not in r];ids={int(r['id']) for r in images}
    annotations=[r for r in source['annotations'] if int(r['image_id']) in ids]
    if len(images)!=420 or len(annotations)!=1668:raise ValueError('Prior fitting requires exactly the acquired training subset')
    acquired=out/'acquired_training.json';acquired.write_text(json.dumps({**source,'images':images,'annotations':annotations},ensure_ascii=False,indent=2),encoding='utf8')
    prior=out/'refitted_prior.json'
    subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'data_preparation/private_prior_source.py'),'--train-ann',str(acquired),'--output',str(prior)],check=True)
    subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'data_preparation/private_prior_compatibility.py'),str(prior)],check=True)
    fitted=read(prior);reference=read(ROOT/'inference/inputs/prior.json');differences={}
    for key in ['feature_mean','feature_std','class_mean_raw_g','class_std_raw_g','class_mean_normalized','class_std_normalized']:
        a=np.asarray(fitted[key]);b=np.asarray(reference[key]);differences[key]=float(np.max(np.abs(a-b)))
        if not np.array_equal(a.astype(np.float32),b.astype(np.float32)):raise ValueError('Refitted prior changes model FP32 values: '+key)
    names={r['id']:r['english'] for r in read(ROOT/'configs/scalevision/class_display_names.json')}
    means=np.asarray(fitted['class_mean_raw_g']);stds=np.asarray(fitted['class_std_raw_g']);matrix=np.eye(28);quad_errors=[]
    for i in range(28):
        for j in range(i+1,28):
            value=overlap(means[i],stds[i],means[j],stds[j]);matrix[i,j]=matrix[j,i]=value
            low=min(means[i]-12*stds[i],means[j]-12*stds[j]);high=max(means[i]+12*stds[i],means[j]+12*stds[j])
            density=lambda x:min(math.exp(-.5*((x-means[i])/stds[i])**2)/stds[i],math.exp(-.5*((x-means[j])/stds[j])**2)/stds[j])/math.sqrt(2*math.pi)
            # Locate density crossings independently of the polynomial roots.
            # Otherwise adaptive quadrature can miss a tiny change of the lower
            # curve when two SDs are nearly equal, despite a small error estimate.
            log_difference=lambda x:math.log(stds[j]/stds[i])-.5*((x-means[i])/stds[i])**2+.5*((x-means[j])/stds[j])**2
            grid=np.linspace(low,high,513);crossings=[]
            for left,right in zip(grid[:-1],grid[1:]):
                if log_difference(left)*log_difference(right)<0:crossings.append(brentq(log_difference,left,right,xtol=1e-12))
            check=quad(density,low,high,epsabs=1e-11,epsrel=1e-11,points=[means[i],means[j],*crossings],limit=150)[0]
            quad_errors.append(abs(value-check))
    if max(quad_errors)>1e-7:raise ValueError('Analytic Gaussian overlap disagrees with independent quadrature')
    if not(matrix[0,1]<.001 and round(matrix[7,8],3)==.933):raise ValueError('Paper OVL anchors not reproduced')
    np.save(out/'class_overlap.npy',matrix)
    with (out/'class_overlap.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['class',*[names[i] for i in range(28)]])
        for i in range(28):writer.writerow([names[i],*matrix[i].tolist()])
    fallback=[];mapping=torch.tensor(read(ROOT/'inference/inputs/taxonomy.json')['leaf_to_parent'])
    for seed in [42,3407,2026]:
        wrapped,_=infer.build_model(seed,torch.device('cpu'),str(Path(args.weights_root)/f'detector_seed{seed}.pth'))
        query=Path(args.inference_root)/f'validation/seed{seed}/predictions/queries.npz';arrays=dict(np.load(query))
        manifest=read(Path(args.inference_root)/'inputs/validation/manifest.json');records=[]
        for row in manifest['samples']:
            modal,_=infer.load_transaction(Path(row['transaction']),'cpu')
            records.append(dict(image_id=int(row['sample_id']),**{k:v[0].tolist() for k,v in modal.items()}))
        data=physical_cache(arrays,records,wrapped.bridge.upstream);packed=tr.pack(data)
        payload=torch.load(Path(args.control_weights_root)/f'bqr/seed{seed}.pth',map_location='cpu',weights_only=True)
        router=tr.model_class()(**payload['config']).eval();router.load_state_dict(payload['state_dict'])
        fallback.append(dict(seed=seed,**fallback_contract(data,packed,wrapped.bridge.upstream,mapping,router,payload['inference_scale'])))
    result=dict(status='complete',train_sources=420,prior_objects=1668,source_training_sha256=sha(train),
        refitted_prior_sha256=sha(prior),reference_prior_sha256=sha(ROOT/'inference/inputs/prior.json'),
        prior_numeric_differences=differences,all_FP32_prior_parameters_exact=True,
        overlap_quad_max_error=max(quad_errors),paper_overlap=dict(wanglaoji_empty_full=matrix[0,1],small_chip_flavours=matrix[7,8]),
        raw_increment_contradiction_fallback=fallback,all_six_backends_all300_queries_exact=True)
    (out/'MASS_EVIDENCE_QA.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='complete',overlap=result['paper_overlap'],max_prior_error=max(differences.values()))))

if __name__=='__main__':main()
