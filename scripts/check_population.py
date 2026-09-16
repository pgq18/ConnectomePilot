"""Input sensitivity, sparse numerical agreement and GPU throughput."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy import sparse
from flylab.population import PopulationBrain, DATA

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(torch.device(args.device))
    result={'device':args.device,'throughput':[]}
    for batch in (1,8,32):
        brain=PopulationBrain(batch,args.device)
        x=torch.zeros((batch,23),device=args.device); x[:,:9]=.6; x[:,19]=1
        for _ in range(4): brain.advance(x)
        torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(20): brain.advance(x)
        torch.cuda.synchronize()
        result['throughput'].append({'batch':batch,'environment_steps_per_second':batch*20/(time.perf_counter()-start),
             'gpu_allocated_mib':torch.cuda.memory_allocated()/2**20})
        del brain; gc.collect(); torch.cuda.empty_cache()
    brain=PopulationBrain(6,args.device,noise_hz=0)
    result['model']=brain.info()
    matrix=sparse.load_npz(DATA/'weights.npz')
    rng=np.random.default_rng(193)
    s=(rng.random((brain.n,6)) < .03).astype(np.float32)
    actual=torch.sparse.mm(brain.weights,torch.as_tensor(s,device=args.device)).cpu().numpy()
    expected=matrix@s
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    result['sparse_max_absolute_error']=float(np.max(np.abs(actual-expected)))
    x=torch.zeros((6,23),device=args.device); x[:,:9]=1; x[:,19]=1
    x[1,0:3]=.2; x[2,3:6]=.2; x[3,6:9]=.2
    x[4:6,3:6]=.5; x[4,12:15]=1; x[5,12:15]=-1
    for _ in range(15): features=brain.advance(x)
    z=features[:,:2*len(brain.dn)].cpu().numpy()
    result['sensitivity']={'conditions':['empty','right','front','left','closing','receding'],
        'active_dn_per_condition':(z[:,:len(brain.dn)]>1e-6).sum(1).tolist(),
        'feature_l2_distance':np.linalg.norm(z[:,None,:]-z[None,:,:],axis=-1).tolist(),
        'noise_hz':0, 'neural_seconds':15*.1}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
