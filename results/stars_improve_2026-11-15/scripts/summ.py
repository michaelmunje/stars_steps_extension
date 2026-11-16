import json,sys
d=json.load(open(sys.argv[1])); names=sys.argv[2].split(',')
cols=['ade','fde','mean_ade','minade','joint_ade','speed','mean_speed','speed_ratio_moving','speed_sample_w1','turn','pair0.2','allk_pair0.2','pair0.4','allk_pair0.4','allk_new_pair0.4','mean_pair0.4','shuffled_pair0.4','allk_pairstep0.4','diversity_final']
for ds,r in d['metrics'].items():
    print(f"== {ds}: gt speed {r.get('speed_gt',0):.3f} gt_pair0.2 {r.get('gt_pair0.2',0):.4f} gt_pair0.4 {r.get('gt_pair0.4',0):.4f} gt_pairstep0.4 {r.get('gt_pairstep0.4',0):.4f} stationary {r.get('stationary_gt',0):.3f}")
    print('model'.ljust(16)+''.join(c[:9].rjust(10) for c in cols))
    for n in names:
        def g(c):
            k = f"speed_{n}_w1" if c=='speed_sample_w1' else f"{n}_{c}"
            return r.get(k,float('nan'))
        print(n.ljust(16)+''.join(f"{g(c):10.4f}" for c in cols))
