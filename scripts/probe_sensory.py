"""Linear decoding of old frozen LIF populations on held-out teacher episodes."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flylab.population import PopulationBrain
from scripts.train_population import write_json


@torch.no_grad()
def features(brain, path):
    data = np.load(path); offsets = data['offsets']
    rows, targets, context = [], [], []
    for begin in range(0, len(offsets)-1, brain.batch):
        episodes = [data['observations'][offsets[i]:offsets[i+1]]
                    for i in range(begin, min(begin+brain.batch, len(offsets)-1))]
        brain.reset()
        for t in range(max(map(len, episodes))):
            raw = np.zeros((brain.batch,23), np.float32)
            valid = []
            for i, episode in enumerate(episodes):
                if t < len(episode): raw[i] = episode[t]; valid.append(i)
            encoded = brain.advance(torch.as_tensor(raw,device=brain.device))
            if t % 3 == 0:
                rows.append(encoded[valid,:-5]); targets.append(torch.as_tensor(raw[valid,:18],device=brain.device))
                context.append(torch.as_tensor(raw[valid,18:],device=brain.device))
    return torch.cat(rows), torch.cat(targets), torch.cat(context)


def fit_predict(train_x, train_y, test_x):
    mean = train_x.mean(0); scale = train_x.std(0).clamp_min(.01)
    a = torch.cat([(train_x-mean)/scale,torch.ones(len(train_x),1,device=train_x.device)],1)
    b = torch.cat([(test_x-mean)/scale,torch.ones(len(test_x),1,device=test_x.device)],1)
    regularizer = torch.eye(a.shape[1],device=a.device)*10;regularizer[-1,-1]=0
    weights = torch.linalg.solve(a.T@a+regularizer,a.T@train_y)
    return b@weights


def metrics(prediction, target):
    error = (prediction-target).square().mean(0)
    r2 = 1-error/target.var(0,unbiased=False).clamp_min(1e-8)
    valid = target[:,:9].min(1).values < 2.9/3
    return {'distance_rmse_m':float(error[:9].mean().sqrt()*3),
            'closing_speed_rmse_m_per_s':float(error[9:].mean().sqrt()*3),
            'distance_mean_r2':float(r2[:9].mean()),'closing_speed_mean_r2':float(r2[9:].mean()),
            'nearest_bearing_accuracy':float((prediction[valid,:9].argmin(1)==target[valid,:9].argmin(1)).float().mean()),
            'r2_per_channel':r2.tolist(),'samples':len(target)}


def main():
    torch.set_num_threads(1);torch.cuda.set_device(1)
    root=ROOT/'results/learning';results={}
    for wiring in ('real','shuffled'):
        brain=PopulationBrain(8,'cuda:1',wiring=wiring,seed=800)
        x,y,c=features(brain,root/'train.npz');xt,yt,ct=features(brain,root/'validation.npz')
        results[wiring]=metrics(fit_predict(x,y,xt),yt)
        results['context_only']=metrics(fit_predict(c,y,ct),yt)
        results['raw_sensor_control']=metrics(fit_predict(y,y,yt),yt)
        print(wiring,results[wiring],flush=True)
        del brain,x,y,c,xt,yt,ct;torch.cuda.empty_cache()
    write_json(root/'sensory-probe.json',{'ridge_lambda':10,'stride':3,'train_episodes':128,
              'validation_episodes':32,'dataset_split':'by whole episode, no random-frame split',
              'neural_noise_hz':1.2,'results':results})


if __name__=='__main__':main()
