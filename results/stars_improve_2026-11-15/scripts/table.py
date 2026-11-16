"""Compact val table: macro over datasets (excluding 'all') + pooled 'all' collisions."""
import json,sys,glob,os
cols=['ade','minade','joint_ade','netspeed','netspeed_ratio_moving','allk_pair0.2','allk_pair0.4','allk_new_pair0.4','shuffled_pair0.4']
print('run'.ljust(26)+''.join(c[:12].rjust(13) for c in ['macro_'+c for c in ['ade','minade','joint']]+['all_'+c for c in cols[3:]]))
for f in sys.argv[1:]:
    d=json.load(open(f)); m=d['metrics']; name=list(d['checkpoints'])[0] if d['checkpoints'] else 'cv'
    ds=[k for k in m if k!='all']
    macro=lambda c: sum(m[k][f'{name}_{c}'] for k in ds)/len(ds)
    row=[macro('ade'),macro('minade'),macro('joint_ade')]+[m['all'][f'{name}_{c}'] for c in cols[3:]]
    print(os.path.basename(f)[:-5].ljust(26)+''.join(f'{v:13.4f}' for v in row))
