# compute_norms_filtered.py
import glob, os, torch, numpy as np, json

pdir = r'C:\Users\Egor\VsCode project\Scene_graph_localization\res'
files = sorted(glob.glob(os.path.join(pdir, '*.pt')))

# Укажи индексы, которые нужно НОРМАЛИЗОВАТЬ (0-based)
# Например для node: пропускаем class_idx (0) и visible (12) -> нормализуем остальные
node_cont_indices = [1,2,3,4,5,6,7,8,9,10,11,13]  # скорректируй под фактический порядок
# для edge_attr: пропустим последний (label_idx)
edge_cont_indices = [0,1,2,3,4,5,6,7]

node_sum = None; node_sq = None; node_count = 0
edge_sum = None; edge_sq = None; edge_count = 0

for f in files:
    d = torch.load(f, weights_only=False)
    x = d['x'].numpy()  # [N, F]
    if x.shape[0] > 0:
        xc = x[:, node_cont_indices]  # выбираем только контин. признаки
        if node_sum is None:
            node_sum = xc.sum(axis=0); node_sq = (xc**2).sum(axis=0)
        else:
            node_sum += xc.sum(axis=0); node_sq += (xc**2).sum(axis=0)
        node_count += xc.shape[0]
    ea = d.get('edge_attr')
    if ea is not None and ea.shape[0] > 0:
        ea = ea.numpy()
        ec = ea[:, edge_cont_indices]
        if edge_sum is None:
            edge_sum = ec.sum(axis=0); edge_sq = (ec**2).sum(axis=0)
        else:
            edge_sum += ec.sum(axis=0); edge_sq += (ec**2).sum(axis=0)
        edge_count += ec.shape[0]

norms = {}
if node_count>0:
    mean = (node_sum / node_count).tolist()
    var = (node_sq / node_count - np.array(mean)**2).clip(min=1e-12)
    std = np.sqrt(var).tolist()
    norms['node_mean'] = mean
    norms['node_std'] = std
    norms['node_cont_indices'] = node_cont_indices

if edge_count>0:
    emean = (edge_sum / edge_count).tolist()
    evar = (edge_sq / edge_count - np.array(emean)**2).clip(min=1e-12)
    estd = np.sqrt(evar).tolist()
    norms['edge_mean'] = emean
    norms['edge_std'] = estd
    norms['edge_cont_indices'] = edge_cont_indices

out = os.path.join(pdir, 'norms.json')
with open(out, 'w', encoding='utf-8') as fh:
    json.dump(norms, fh, indent=2)
print("Saved norms:", out)