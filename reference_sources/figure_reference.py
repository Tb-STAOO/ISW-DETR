"""Functions preserved from the original experiment implementation."""
from pathlib import Path
import numpy as np
CAPTIONS={}

def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white',
        'axes.linewidth':1.2,'grid.linewidth':1.2,'lines.linewidth':1.2,'patch.linewidth':1.2,
        'xtick.major.width':1.2,'xtick.minor.width':1.2,'ytick.major.width':1.2,'ytick.minor.width':1.2})
    return plt


def save(fig,name,caption):
    import matplotlib.pyplot as plt
    # Enforce stroke width after all artists and legend copies exist. 1.2 pt
    # remains >=1 pt at the approximate 90% manuscript embedding scale.
    fig.canvas.draw()
    for artist in fig.findobj():
        get=getattr(artist,'get_linewidth',None);set_=getattr(artist,'set_linewidth',None)
        if callable(get) and callable(set_):
            width=get()
            if np.isscalar(width) and width>0:set_(max(1.2,float(width)))
        get=getattr(artist,'get_linewidths',None);set_=getattr(artist,'set_linewidths',None)
        if callable(get) and callable(set_):
            widths=np.asarray(get(),dtype=float)
            if widths.size:set_(np.where(widths>0,np.maximum(widths,1.2),widths))
        get=getattr(artist,'get_markeredgewidth',None);set_=getattr(artist,'set_markeredgewidth',None)
        if callable(get) and callable(set_):set_(max(1.2,float(get())))
    for ext in ('pdf','svg','png'):fig.savefig(FIG/f'{name}.{ext}',dpi=300,facecolor='white')
    CAPTIONS[name]=caption
    plt.close(fig)


def panel(ax,letter,title):
    ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold',pad=9)


def figure_system():
    plt=plotting();from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    fig,ax=plt.subplots(figsize=(7.2,4.2));fig.subplots_adjust(left=.025,right=.99,bottom=.03,top=.98)
    ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.1',facecolor=color,edgecolor='#45505A',linewidth=.9))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
    def arrow(a,b,text=None,style='-',color='#45505A'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,linewidth=1,color=color,linestyle=style,connectionstyle='arc3'))
        if text:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,text,ha='center',fontsize=8,color=color)
    box(.1,4.4,1.2,.75,'RGB\nimage','#EEF3F8')
    box(1.7,4.1,3.1,1.35,'RT-DETRv2 + auxiliary I–S\nPResNet-18 · HybridEncoder\n300 queries: logits + boxes','#E0EDF7')
    box(5.3,4.1,2.15,1.35,'Visual candidates\noriginal score order\nIoU filter · E + 12','#EEF3F8')
    box(.1,1.65,1.2,1,'Mass\nrecords','#FFF3DD')
    box(1.7,1.25,3.1,1.8,'Transaction verification\nrecord · chain · total · endpoint\nTraining-only mass prior\nscene support + record residual','#FFF0D5')
    box(5.3,1.35,2.15,1.55,'State-controlled\nassociation\nmean grouped logits\none-to-one assignment','#E4F2EA')
    box(7.95,3.05,1.85,1.65,'Partial-contrast\nclassification\nhalf visual contrast\n+ mass evidence','#E4F2EA')
    box(7.95,.35,1.85,1.1,'Detections\nquery boxes retained','#EEF3F8')
    arrow((1.3,4.78),(1.7,4.78));arrow((4.8,4.78),(5.3,4.78))
    arrow((1.3,2.15),(1.7,2.15));arrow((4.8,2.15),(5.3,2.15))
    arrow((6.37,4.1),(6.37,2.9));arrow((7.45,2.12),(8.2,3.05))
    arrow((7.45,4.78),(8.2,4.7));arrow((8.88,3.05),(8.88,1.45))
    ax.plot([3.25,3.25,7.72,7.72,7.95],[4.1,3.6,3.6,.89,.89],color='#0072B2',linestyle='--',linewidth=1)
    ax.text(4.05,3.68,'Original visual boxes',fontsize=8,color='#0072B2')
    ax.plot([3.25,3.25,7.95],[1.25,.15,.15],color='#88651D',linestyle=':',linewidth=1)
    ax.text(4.9,.28,'Missing / rejected transaction → visual output',ha='center',fontsize=8,color='#88651D')
    save(fig,'Fig1_system','ISW-DETR overview. Visual detection supplies candidate query logits and boxes. Verified mass evidence and grouped-mean logits determine record association; partial visual state contrast is retained in final classification. Candidate selection precedes the state transformation. Query box coordinates are retained. Missing or rejected transactions return the original visual output.')


def figure_acquisition():
    plt=plotting();from PIL import Image;from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    prior=read(PACKAGE/'inputs/prior.json');ann=read(PACKAGE/'inputs/holdout.json')
    im=ann['images'][0];anns=sorted([a for a in ann['annotations'] if a['image_id']==im['id']],key=lambda a:a['placement_order'])
    imagepath=HOLDOUT_IMAGES/im['file_name'];SOURCE[str(imagepath)]=sha(imagepath)
    scene=np.asarray(Image.open(imagepath).convert('RGB'))
    fig=plt.figure(figsize=(7.2,5.7));gs=fig.add_gridspec(2,2,left=.11,right=.98,bottom=.10,top=.94,hspace=.42,wspace=.3)
    a=fig.add_subplot(gs[0,0]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');panel(a,'a','Vision–weight acquisition')
    nodes=[(.02,.67,.29,.20,'6-kg load cell\nLAB-B-B'),(.38,.67,.25,.20,'CMCU-07\nacquisition'),
        (.70,.67,.28,.20,'STM32\nforwarder'),(.02,.22,.29,.20,'RGB camera\n1280 × 720'),
        (.45,.22,.53,.20,'Host transaction builder\nimage + ordered mass records')]
    for x,y,w,h,t in nodes:
        a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',facecolor='#EEF3F8',edgecolor='#45505A',linewidth=.8));a.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=7)
    for start,end in [((.31,.77),(.38,.77)),((.63,.77),(.70,.77)),((.84,.65),(.84,.44)),((.31,.32),(.43,.32))]:
        a.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=10,color='#45505A',linewidth=.8))
    a.text(.84,.54,'USB',ha='center',fontsize=7)
    b=fig.add_subplot(gs[0,1]);b.imshow(scene);b.axis('off');panel(b,'b','Measured scene')
    c=fig.add_subplot(gs[1,0]);mass=[anns[0]['weight_features']['stable_before_g']]+[x['weight_features']['stable_after_g'] for x in anns]
    c.step(np.arange(len(mass)),mass,where='post',color='#0072B2',linewidth=1.5);c.scatter(np.arange(len(mass)),mass,color='#0072B2',s=20)
    c.axhline(im['scene_total_weight_g'],linestyle=':',color='#555555',linewidth=1,label='Final measured total')
    c.set(xlabel='Completed placement',ylabel='Stable plateau mass (g)',xticks=range(len(mass)),ylim=(-20,max(mass)*1.15));c.legend(loc='upper left',frameon=False,fontsize=7)
    panel(c,'c','Recorded mass plateaus')
    sg=gs[1,1].subgridspec(2,1,hspace=.70)
    names=read(CLASS_NAMES_PATH);colors=['#0072B2','#D55E00']
    for j,(pair,xlim,title) in enumerate([((0,1),(0,370),'Same product: empty / full'),((7,8),(34,45),'Similar packages: two flavours')]):
        ax=fig.add_subplot(sg[j]);x=np.linspace(*xlim,800)
        for k,idx in enumerate(pair):
            mu=prior['class_mean_raw_g'][idx];sd=prior['class_std_raw_g'][idx]
            y=np.exp(-.5*((x-mu)/sd)**2)/(sd*np.sqrt(2*np.pi));ax.plot(x,y,color=colors[k],linestyle=['-','--'][k],label=names[idx]['english'],linewidth=1.3)
        ax.set(xlim=xlim,ylabel='Density');ax.legend(fontsize=6.6,frameon=False,loc='upper right')
        if j==0:panel(ax,'d','Training mass distributions')
        else:ax.set_xlabel('Mass (g)')
    save(fig,'Fig2_acquisition','Acquisition and representative physical evidence. (a) Hardware and host transaction chain, redrawn from the existing documented acquisition setup. (b) The first measured-scene image by annotation image ID. (c) Recorded stable plateau endpoints for its ordered placements, rather than a continuous sensor waveform; the dotted line marks the independently logged final total. (d) Training-only Gaussian mass parameters illustrate an empty/full pair and a near-overlapping flavour pair; the class standard-deviation floor is retained.')


def figure_effects():
    plt=plotting();bs=read(BOOTSTRAP_SUMMARY)
    alpha=read(VALIDATION_ANALYSIS/'REASSOCIATED_SCREEN_RESULTS.json')['rows'];sessions=read(DIAGNOSTICS/'SESSION_RESULTS.json')['rows']
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.9));fig.subplots_adjust(left=.12,right=.98,bottom=.25,top=.87,wspace=.65)
    a,b,c=axs
    labels=['vs BQR','vs original\nassociation','vs max']
    for i,row in enumerate(bs['contrasts']):
        v=row['point_gain_ap_points'];lo,hi=row['percentile95_ci_ap_points']
        a.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#0072B2',capsize=3)
    a.axvline(0,color='#777777',linewidth=.8);a.set(yticks=range(3),yticklabels=labels,xlabel='Measured ΔAP (points)');a.invert_yaxis();panel(a,'a','Conditional gains')
    xx=np.array([0,.5,1.]);vv=[]
    for seed in SEEDS:
        yy=[next(r['ap']*100 for r in alpha if r['seed']==seed and r['alpha']==x and r['gate']==1.) for x in xx]
        b.plot(xx,yy,color='#0072B2',alpha=.3,linewidth=.8);vv.append(yy)
    b.plot(xx,np.mean(vv,axis=0),'o-',color='#0072B2',linewidth=1.5);b.set(xticks=xx,xlabel='Readout contrast removal α',ylabel='Validation AP');panel(b,'b','Validation setting')
    allv=[]
    for seed in SEEDS:
        yy=[]
        for s in ('session_01','session_02','session_03'):
            vals={r['method']:r['ap'] for r in sessions if r['seed']==seed and r['session']==s}
            yy.append(100*(vals[MAIN]-vals['original_BQR']))
        c.plot([1,2,3],yy,'o-',color='#009E73',alpha=.3,linewidth=.8,markersize=3);allv.append(yy)
    c.plot([1,2,3],np.mean(allv,axis=0),'D-',color='#009E73',linewidth=1.5,markersize=4)
    c.axhline(0,color='#777777',linewidth=.8);c.set(xticks=[1,2,3],xlabel='Acquisition session',ylabel='ΔAP vs BQR (points)');panel(c,'c','Measured sessions')
    save(fig,'Fig3_effects','Evidence for the selected state-contrast design. (a) Mean measured AP differences and pointwise paired-scene 95% percentile intervals from 10,000 shared draws. (b) Validation AP at three readout-removal settings with mean association and unchanged physical coefficients; the selected setting is 0.5. (c) Measured-session AP gains over BQR. Thin lines denote individual trained seeds and thick lines their mean in (b,c); they are not confidence intervals.')


def figure_stress():
    plt=plotting();vis=read(VALIDATION_ANALYSIS/'VISUAL_STRESS_RESULTS.json')['rows'];phys=read(VALIDATION_ANALYSIS/'PHYSICAL_STRESS_RESULTS.json')['rows']
    methods=['original_BQR',MAIN,'half_readout_max_assignment','full_mean_erasure']
    names=['BQR','Mean + half (ISW-DETR)','Max + half','Mean + full']
    colors=['#777777','#0072B2','#D55E00','#009E73'];marks=['o','s','^','D']
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.2));fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.93,hspace=.55,wspace=.35)
    def curve(ax,rows,xs,key,conditions):
        for m,name,col,mk in zip(methods,names,colors,marks):
            vv=[]
            for seed in SEEDS:
                yy=[next(r['ap']*100 for r in rows if r['seed']==seed and r['method']==m and r[key]==co) for co in conditions]
                ax.plot(xs,yy,color=col,alpha=.15,linewidth=.7);vv.append(yy)
            ax.plot(xs,np.mean(vv,axis=0),marker=mk,color=col,linewidth=1.4,markersize=4,label=name)
        ax.set_ylabel('Validation AP');ax.set_xticks(xs)
    a,b,c,d=axs.ravel()
    curve(a,vis,[0,1,2,4],'condition',['normal','contrast_delta1','contrast_delta2','contrast_delta4']);a.set_xlabel('State-contrast amplitude δ (logit)');panel(a,'a','Opposed state perturbation')
    for j,(m,name,col,mk) in enumerate(zip(methods,names,colors,marks)):
        vals=[]
        for seed in SEEDS:
            normal=next(r['ap']*100 for r in vis if r['seed']==seed and r['method']==m and r['condition']=='normal')
            yy=[next(r['ap']*100 for r in vis if r['seed']==seed and r['method']==m and r['condition']==co)-normal for co in ('contrast_delta2','pair_common_delta2')]
            b.plot(np.array([0,1])+(j-1.5)*.07,yy,color=col,alpha=.22,linewidth=.7);vals.append(yy)
        b.plot(np.array([0,1])+(j-1.5)*.07,np.mean(vals,axis=0),marker=mk,color=col,linewidth=1.4,markersize=4)
    b.axhline(0,color='#777777',linewidth=.8);b.set(xticks=[0,1],xticklabels=['Opposed','Common'],ylabel='ΔAP from clean (points)',xlabel='Equal nominal L2 perturbation (δ = 2)');panel(b,'b','Direction control')
    curve(c,phys,[0,.25,1,5],'sigma_g',[0.,.25,1.,5.]);c.set_xticks([0,1,5]);c.set_xlabel('Interior plateau offset σ (g)');panel(c,'c','Conserved total, shifted records')
    curve(d,vis,[0,.5,1],'condition',['likelihood_temperature0','likelihood_temperature0.5','normal']);d.set_xlabel('Mass log-likelihood scale τ');panel(d,'d','Mass-evidence flattening')
    handles,labels=a.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.01))
    save(fig,'Fig4_stress','Controlled evidence interventions on the 90 validation scenes. (a,b) Opposed full/empty-logit changes are compared with pair-common changes of the same nominal per-query L2 magnitude. (c) Synthetic interior-plateau offsets redistribute event mass while preserving the original endpoint totals and checks. (d) The class log likelihood is multiplied by scale τ before evidence reconstruction; τ=0 removes class-specific mass information. Curves show three-seed means, with faint individual-seed trajectories. Candidate pools, boxes and coefficients are fixed. These interventions isolate evidence responses rather than measured RGB or sensor-noise distributions.')


def figure_classes(diag):
    plt=plotting();from PIL import Image;from matplotlib.patches import Rectangle
    names=read(CLASS_NAMES_PATH)
    fig=plt.figure(figsize=(7.2,8.5));gs=fig.add_gridspec(3,2,left=.28,right=.98,bottom=.10,top=.96,width_ratios=[1.4,1],height_ratios=[1.3,1,1],wspace=.42,hspace=.65)
    a=fig.add_subplot(gs[:,0]);delta={}
    for split,col,mk,off,label in [('internal','#0072B2','o',-.12,'Internal'),('holdout','#D55E00','s',.12,'Measured')]:
        new=np.mean([r['class_ap_points'] for r in diag['per_class'] if r['split']==split and r['method']==MAIN],axis=0)
        old=np.mean([r['class_ap_points'] for r in diag['per_class'] if r['split']==split and r['method']=='original_BQR'],axis=0)
        delta[split]=(new-old).tolist();a.scatter(new-old,np.arange(28)+off,s=13,color=col,marker=mk,label=label)
    a.axvline(0,color='#777777',linewidth=.8);a.set(yticks=range(28),yticklabels=[r['english'] for r in names],xlabel='Δ class AP vs BQR (points)');a.invert_yaxis();a.tick_params(axis='y',labelsize=7);a.grid(axis='x',alpha=.18);a.legend(frameon=False,fontsize=7,loc='lower right');panel(a,'a','All 28 classes')
    b=fig.add_subplot(gs[0,1]);rows=[r for r in diag['corrections'] if r['split']=='holdout']
    corrected=[sum(r['identity_corrected'] for r in rows),sum(r['state_corrected'] for r in rows)]
    harmed=[sum(r['identity_harmed'] for r in rows),sum(r['state_harmed'] for r in rows)]
    xx=np.arange(2);b.bar(xx-.17,corrected,.32,color='#0072B2',label='Corrected');b.bar(xx+.17,harmed,.32,color='#D55E00',label='Harmed')
    for off,vv in [(-.17,corrected),(.17,harmed)]:
        for x,v in zip(xx,vv):b.text(x+off,v+.5,str(v),ha='center',fontsize=8)
    b.set(xticks=xx,xticklabels=['Identity','State only'],ylabel='Fixed-query observations');b.legend(frameon=False,fontsize=7);b.set_ylim(0,max(corrected)*1.23);panel(b,'b','Measured class changes')
    for j,case in enumerate(diag['selected_cases']):
        ax=fig.add_subplot(gs[j+1,1]);ip=HOLDOUT_IMAGES/case['file_name'];SOURCE[str(ip)]=sha(ip)
        im=Image.open(ip).convert('RGB');w,h=im.size;cx,cy,bw,bh=case['box_cxcywh'];x1=(cx-bw/2)*w;y1=(cy-bh/2)*h;x2=(cx+bw/2)*w;y2=(cy+bh/2)*h
        pad=.18*max(x2-x1,y2-y1);crop=(max(0,int(x1-pad)),max(0,int(y1-pad)),min(w,int(x2+pad)),min(h,int(y2+pad)))
        ax.imshow(im.crop(crop));ax.add_patch(Rectangle((x1-crop[0],y1-crop[1]),x2-x1,y2-y1,fill=False,color='#0072B2',linewidth=1));ax.axis('off')
        title=f"{case['kind'].capitalize()} (image {case['image_id']})"
        if j==0:panel(ax,'c',title)
        else:panel(ax,'d',title)
        compact=lambda idx:names[idx]['english'].replace('Canned ','').replace(' (empty)',' empty').replace('Large ','L. ').replace('Small ','S. ')
        ax.text(.5,-.08,f"RT-DETRv2: {compact(case['original_rtdetr_argmax'])}\nBQR: {compact(case['before'])}\nISW: {compact(case['after'])}\nGT: {compact(case['truth'])}",transform=ax.transAxes,ha='center',va='top',fontsize=7)
    write(DATA/'CLASS_AP_CHANGES.json',delta)
    save(fig,'Fig5_classes','Class-level behavior relative to BQR. (a) Changes in native COCO AP for every class, averaged over three seeds. (b) Corrections and harms under the fixed original geometric query-to-GT mapping on measured scenes. Counts pool repeated observations across three trained seeds; state-only changes keep both predicted identities equal to the true identity. (c,d) Crops show the first corrected and first harmed seed42 measured examples by image ID then query ID. Original RT-DETRv2, BQR and ISW-DETR labels use the same query IDs and bitwise-identical box coordinates in these two cases. These are diagnostic argmax labels rather than the thresholded detection display.')

