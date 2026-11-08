"""Current manuscript table geometry/maxima; no stored AP values."""
import re
from decimal import Decimal

CENTER = r'>{\centering\arraybackslash}'

def table_shell(header, body, specification, padding=4):
    return ('\\begingroup\n\\setlength{\\tabcolsep}{'+str(padding)+'pt}\n'
            '\\renewcommand{\\tabularxcolumn}[1]{m{#1}}\n'
            '\\begin{tabularx}{\\linewidth}{'+specification+'}\n\\toprule\n'+header+
            '\n\\midrule\n'+body.strip()+'\n\\bottomrule\n\\end{tabularx}\n\\endgroup\n')

def base_transform(text, kind, language):
    """R064-R067: same immutable cells, updated centered display structure."""
    if kind == 'main':
        first=(r'\multirow{2}{*}[-2pt]{Method} & \multirow{2}{*}[-2pt]{Input} & Internal & \multicolumn{4}{c}{Measured transactions} \\'
               if language=='English' else
               r'\multirow{2}{*}[-2pt]{方法} & \multirow{2}{*}[-2pt]{输入} & 内部 & \multicolumn{4}{c}{实测事务} \\')
        second=r'& & AP & AP & AP$_{50}$ & AP$_{75}$ & AR$_{100}$ \\'
        header=first+'\n'+r'\cmidrule(lr){3-3}\cmidrule(lr){4-7}'+'\n'+second
        body=text.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
        specification='@{}'+CENTER+r'm{0.33\linewidth}'+CENTER+r'm{0.14\linewidth}*{5}{'+CENTER+'X}@{}'
        return table_shell(header,body,specification,4)
    if kind == 'ablation':
        blocks=text.split(r'\midrule')
        body_a=blocks[1].strip()
        body_b=blocks[-1].split(r'\bottomrule',1)[0]
        converted=[]
        for line in body_b.splitlines():
            if not line.strip():continue
            match=re.fullmatch(r'\\multicolumn\{4\}\{@\{\}l\}\{(.*)\}\s*&\s*(.*?)\s*\\\\',line.strip())
            assert match,line
            converted.append(match[1]+' & '+match[2]+r' \\')
        if language=='English':
            head_a=r'Association & Visual score mapping & I--S & Internal AP (\%) & Measured AP (\%) \\'
            head_b=r'Control & Validation AP (\%) \\'
        else:
            head_a=r'关联 & 视觉评分映射 & I--S & 内部 AP（\%） & 实测 AP（\%） \\'
            head_b=r'对照 & 验证 AP（\%） \\'
        spec_a='@{}'+CENTER+r'm{0.18\linewidth}'+CENTER+r'm{0.31\linewidth}'+CENTER+r'm{0.08\linewidth}*{2}{'+CENTER+'X}@{}'
        # Equal table edges and a shared right AP-column center for both panels.
        spec_b='@{}'+CENTER+r'm{\dimexpr0.785\linewidth+2\tabcolsep\relax}'+CENTER+'X@{}'
        marker=lambda letter:r'\noindent\makebox[\linewidth][l]{\textbf{('+letter+r')}}\par\nobreak\vspace{2pt}'+'\n'
        return (marker('a')+table_shell(head_a,body_a,spec_a,4)+
                r'\par\nobreak\vspace{6pt}'+'\n'+marker('b')+
                table_shell(head_b,'\n'.join(converted),spec_b,4))
    assert kind=='cost'
    labels=(['Path','Parameters (M)','Median (ms)','IQR (ms)','P95 (ms)','GPU increment (MiB)']
            if language=='English' else ['路径','参数量（M）','中位数（ms）','IQR（ms）','P95（ms）','GPU增量（MiB）'])
    header=' & '.join(r'\mbox{'+label+'}' for label in labels)+r' \\'
    body=text.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
    specification='@{}'+CENTER+r'm{0.13\linewidth}*{5}{'+CENTER+'X}@{}'
    return table_shell(header,body,specification,2)

def highlight_maxima(text, kind, language):
    """R043: highest displayed scalar per metric/group, including all displayed ties."""
    values = {}  # Recomputed tables contain literal numeric cells.
    lines=text.splitlines(); cells=[]; group=0
    for line_id,line in enumerate(lines):
        if kind=='ablation' and (r'\textbf{B.' in line or r'\textbf{(b)}' in line):group=1
        if '&' not in line or not line.rstrip().endswith(r'\\'):continue
        row=line.rstrip()[:-2].split('&')
        for col,cell in enumerate(row):
            raw=cell.strip()
            raw=re.sub(r'\\(?:mathbf|textbf)\{([^{}]+)\}',r'\1',raw).strip('$ ')
            expanded=re.sub(r'\\(\w+)',lambda m:values.get(m[1],m[0]),raw)
            if re.fullmatch(r'\d+(?:\.\d+)?',expanded):
                cells.append(dict(line=line_id,col=col,group=group,value=Decimal(expanded),raw=raw,math=cell.strip().startswith('$')))
    maxima={}
    for c in cells:
        key=(c['group'],c['col']);maxima[key]=max(maxima.get(key,c['value']),c['value'])
    reports=[]
    for c in cells:
        row=lines[c['line']].rstrip()[:-2].split('&');selected=c['value']==maxima[c['group'],c['col']]
        clean=c['raw'];body=(r'\mathbf{'+clean+'}') if c['math'] else (r'\textbf{'+clean+'}')
        if not selected:body=clean
        row[c['col']]=' '+('$'+body+'$' if c['math'] else body)+' '
        lines[c['line']]='&'.join(row).rstrip()+r' \\'
        reports.append(dict(group=c['group'],line=c['line'],row_label=row[0].strip(),column=c['col'],value=str(c['value']),bold=selected))
    return '\n'.join(lines)+ ('\n' if text.endswith('\n') else ''),reports
