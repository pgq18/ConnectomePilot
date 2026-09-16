"""Extend the completed three-rule trial without changing its learning rules."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flylab.plasticity import PlasticPolicy, PlasticityWorlds, ThreeFactor, GRAPH
from flylab.population_ppo import advantages
from scripts.train_population import summarize, write_json
from flylab.resume import load_learning_state, save_checkpoint, restore_environment


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_sha(model):
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def calibrate(device, output):
    torch.manual_seed(700)
    model = PlasticPolicy('readout').to(device)
    env = PlasticityWorlds(16, 700, list(range(20900000, 20900016)))
    rng = np.random.default_rng(700)
    raw = env.observe(); hidden = model.initial(16)
    starts = torch.ones(16, dtype=torch.bool, device=device)
    total = torch.zeros(len(model.descending), dtype=torch.float64, device=device)
    square = torch.zeros_like(total); count = 0
    for _ in range(256):
        _, _, hidden, _ = model(torch.as_tensor(raw, device=device), hidden, starts)
        values = hidden[:, model.descending].double()
        total += values.sum(0); square += values.square().sum(0); count += len(values)
        raw, _, done, _ = env.step(np.tanh(rng.normal(0, .4, (16, 2))))
        starts = torch.as_tensor(done, device=device)
    mean = total / count
    std = (square / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-3)
    torch.save({'mean': mean.float().cpu(), 'std': std.float().cpu(), 'count': count,
                'seed': 700, 'graph_sha256': sha(GRAPH)}, output)
    write_json(output.with_suffix('.json'), {'frames': count, 'seed': 700, 'standard_deviation_floor': .001,
               'std_min': float(std.min()), 'std_median': float(std.median()), 'std_max': float(std.max()),
               'fixed_feature_scale': 'DN / sqrt(number of DN), five context channels unchanged',
               'source': 'random actions in training maps only; no teacher, reward fitting or test maps'})


@torch.no_grad()
def evaluate(model, scenes, condition='clean', batch=32):
    device = model.edge_weight.device; rows = []
    for start in range(0, len(scenes), batch):
        selected = scenes[start:start+batch]
        env = PlasticityWorlds(len(selected), 0, selected, condition)
        raw = env.observe(); hidden = model.initial(len(selected))
        begins = torch.ones(len(selected), dtype=torch.bool, device=device)
        finished = np.zeros(len(selected), bool)
        for _ in range(600):
            mean, _, hidden, _ = model(torch.as_tensor(raw, device=device), hidden, begins)
            raw, _, done, episodes = env.step(mean.tanh().cpu().numpy(), autoreset=False)
            for row in episodes: row.pop('slot'); rows.append(row)
            finished |= done
            if finished.all(): break
            begins = torch.as_tensor(done, device=device)
    assert len(rows) == len(scenes)
    return {'condition': condition, 'summary': summarize(rows), 'episodes': rows}


def run(mode, seed, args):
    out = args.output / f'{mode}-{seed}'
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    source = args.source / f'{mode}-{seed}' / 'final.pt'
    checkpoint = torch.load(source, map_location=args.device, weights_only=True)
    base_steps = checkpoint.get('total_steps', checkpoint['config']['steps'])
    assert checkpoint['config']['mode'] == mode and checkpoint['config']['seed'] == seed
    assert checkpoint['config']['graph_sha256'] == sha(GRAPH)
    for name in ['batch','rollout','minibatch']:
        assert getattr(args,name) == checkpoint['config'][name], name
    torch.manual_seed(seed + 2000000); np.random.seed(seed + 2000000)
    model = PlasticPolicy(mode).to(args.device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4) if mode != 'three_factor' else None
    local = ThreeFactor(model, args.batch) if mode == 'three_factor' else None
    load_learning_state(checkpoint, model, optimizer, local, restore_traces='rollout_state' in checkpoint)
    initial_digest = weights_sha(model)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(mode=mode, seed=seed, output=str(out), base_steps=base_steps,
        source_checkpoint=str(source), source_checkpoint_sha256=sha(source),
        continuation='full_rollout_state' if 'rollout_state' in checkpoint else 'model_and_optimizer_new_episodes',
        sampling_seed=seed+1000000, torch_seed=seed+2000000,
        graph_sha256=sha(GRAPH), initial_state_sha256=initial_digest,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        reward_scale=.1, gaussian_std=.4, fixed_bias=True, dopamine_activity_injection=False,
        learning_rates={'ppo': .0003, 'local_edges': .0001, 'local_actor': .002, 'local_value': .005})
    config['output'] = str(out)
    write_json(out / 'config.json', config)
    validation = list(range(args.validation_start, args.validation_start + args.validation_episodes))
    write_json(out / f'validation-{base_steps}.json', evaluate(model, validation))
    env = PlasticityWorlds(args.batch, seed + 1000000)
    raw = env.observe(); hidden = model.initial(args.batch)
    starts = torch.ones(args.batch, dtype=torch.bool, device=args.device)
    if 'rollout_state' in checkpoint:
        state=checkpoint['rollout_state'];restore_environment(env,state['environment'])
        raw=state['raw'].cpu().numpy();hidden=state['hidden'];starts=state['starts']
        torch.set_rng_state(state['torch_cpu_rng'].cpu())
        torch.cuda.set_rng_state(state['torch_device_rng'].cpu(),args.device)
    del checkpoint
    logs = []; episodes = []; completed = base_steps; training_seconds = 0.
    torch.cuda.reset_peak_memory_stats()
    while completed < args.steps:
        started = time.perf_counter()
        records = []
        deltas = []
        for _ in range(args.rollout):
            obs = torch.as_tensor(raw, device=args.device)
            with torch.no_grad():
                mean, value, next_hidden, features = model(obs, hidden, starts)
                latent = mean + model.std * torch.randn_like(mean)
                logp = model.log_prob(mean, latent)
            next_raw, reward, done, rows = env.step(latent.tanh().cpu().numpy())
            reward = torch.as_tensor(reward, device=args.device)
            done_tensor = torch.as_tensor(done, device=args.device)
            if local:
                with torch.no_grad():
                    _, next_value, _, _ = model(torch.as_tensor(next_raw, device=args.device), next_hidden, done_tensor)
                    delta = local.update(next_hidden, features, mean, value, latent, reward, next_value, done_tensor)
                    deltas.append(delta.square().mean())
            else:
                records.append((obs, hidden, starts, latent, logp, value, reward*.1, done_tensor))
            for row in rows: row['environment_steps'] = completed + args.batch
            episodes.extend(rows)
            raw, hidden, starts = next_raw, next_hidden, done_tensor
            completed += args.batch
        update_stats = {}
        if optimizer:
            obs, states, begins, latents, oldlog, oldvalues, rewards, dones = [torch.stack(a) for a in zip(*records)]
            with torch.no_grad():
                _, last, _, _ = model(torch.as_tensor(raw, device=args.device), hidden, starts)
                adv, returns = advantages(rewards, oldvalues, dones, last)
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            obs = obs.flatten(0, 1); states = states.flatten(0, 1); begins = begins.flatten()
            latents = latents.flatten(0, 1); oldlog = oldlog.flatten(); adv = adv.flatten(); returns = returns.flatten()
            losses = []; kl = []
            for _ in range(4):
                order = torch.randperm(len(obs), device=args.device)
                for index in order.split(args.minibatch):
                    mean, value, _, _ = model(obs[index], states[index], begins[index])
                    logp = model.log_prob(mean, latents[index]); logratio = logp - oldlog[index]
                    ratio = logratio.exp()
                    policy_loss = -torch.minimum(ratio * adv[index], ratio.clamp(.8, 1.2) * adv[index]).mean()
                    # Fixed Gaussian variance makes the unsquashed entropy constant.
                    loss = policy_loss + .5 * (value - returns[index]).square().mean()
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO loss')
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                    optimizer.step(); model.project()
                    losses.append(loss.detach()); kl.append((ratio - 1 - logratio).mean().detach())
            update_stats = {'loss': float(torch.stack(losses).mean()), 'approx_kl': float(torch.stack(kl).mean())}
        else:
            update_stats = {'td_squared': float(torch.stack(deltas).mean())}
        if not all(torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError('Nonfinite parameters')
        recent = [row for row in episodes if row['environment_steps'] > completed-args.batch*args.rollout]
        training_seconds += time.perf_counter()-started
        row = {'mode': mode, 'seed': seed, 'steps': completed, 'seconds': training_seconds,
               'gpu_peak_mib': torch.cuda.max_memory_allocated()/2**20,
               'training_episodes': len(episodes), 'recent': summarize(recent) if recent else None, **update_stats}
        logs.append(row); write_json(out / 'progress.json', row)
        if completed % 16384 == 0 or completed == args.steps: print(json.dumps(row), flush=True)
        if completed in args.milestones or completed == args.steps:
            filename='final.pt' if completed==args.steps else f'checkpoint-{completed}.pt'
            save_checkpoint(out/filename,model,optimizer,local,config,completed,env,raw,hidden,starts)
            score=evaluate(model,validation)
            write_json(out/f'validation-{completed}.json',score)
            print(json.dumps({'validation':mode,'seed':seed,'steps':completed,**score['summary']}),flush=True)
    train_seconds = training_seconds
    write_json(out / 'training.json', {'updates': logs, 'episodes': episodes})
    measurements = {'train_seconds': train_seconds, 'gpu_peak_mib': torch.cuda.max_memory_allocated()/2**20,
        'initial_state_sha256': initial_digest, 'final_state_sha256': weights_sha(model),
        'edges': len(model.source), 'neurons': model.n,
        'edge_abs_change_mean': float((model.edge_weight.detach()-model.initial_weight).abs().mean()),
        'edge_abs_change_max': float((model.edge_weight.detach()-model.initial_weight).abs().max()),
        'changed_edges': int(((model.edge_weight-model.initial_weight).abs()>1e-8).sum()),
        'sign_violations': int(((model.edge_weight*model.sign)<0).sum()),
        'edges_at_zero': int((model.edge_weight==0).sum()),
        'edges_at_bound': int((model.edge_weight.abs()>=1).sum())}
    write_json(out / 'measurements.json', measurements)
    write_json(out / 'complete.json', {'mode': mode, 'seed': seed, **measurements,
               'base_steps':base_steps,'total_steps':completed})
    print(json.dumps({'training_complete':str(out),'steps':completed}),flush=True)


def final_evaluations(args):
    """Only after every training run ends: evaluate old and final on new maps."""
    fresh=list(range(args.test_start,args.test_start+args.test_episodes))
    legacy=list(range(5300000,5300100))
    for seed in args.seeds:
        for mode in args.modes:
            out=args.output/f'{mode}-{seed}'
            for stage,path in [('before',args.source/f'{mode}-{seed}/final.pt'),('after',out/'final.pt')]:
                checkpoint=torch.load(path,map_location=args.device,weights_only=True)
                model=PlasticPolicy(mode).to(args.device);model.load_state_dict(checkpoint['model'])
                score=evaluate(model,fresh)
                write_json(out/f'test-{stage}.json',score)
                if stage=='after':
                    write_json(out/'legacy-evaluation.json',[evaluate(model,legacy,c) for c in args.conditions])
                print(json.dumps({'test':mode,'seed':seed,'stage':stage,**score['summary']}),flush=True)
                del model,checkpoint
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, default=ROOT/'results/plasticity/main')
    p.add_argument('--modes', nargs='+', default=['readout', 'ppo_edges', 'three_factor'])
    p.add_argument('--seeds', type=int, nargs='+', default=[71,72,73])
    p.add_argument('--steps', type=int, default=262144)
    p.add_argument('--milestones',type=int,nargs='+',default=[131072,262144])
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--rollout', type=int, default=128)
    p.add_argument('--minibatch', type=int, default=64)
    p.add_argument('--validation-start',type=int,default=5400000)
    p.add_argument('--validation-episodes',type=int,default=100)
    p.add_argument('--test-start',type=int,default=5500000)
    p.add_argument('--test-episodes',type=int,default=200)
    p.add_argument('--skip-final-evaluations',action='store_true')
    p.add_argument('--conditions', nargs='+', default=['clean','noise','delay'])
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if not torch.cuda.is_available(): p.error('This runner requires the authorized workstation GPU.')
    if args.steps % (args.batch*args.rollout): p.error('Steps must divide into full rollouts.')
    if any(x%(args.batch*args.rollout) for x in args.milestones): p.error('Milestones must divide into full rollouts.')
    if not 5400000 <= args.validation_start < 5500000 or not 5500000 <= args.test_start < 5600000:
        p.error('Use the reserved validation/test ranges.')
    if args.output.exists():p.error('Use a new output directory.')
    torch.set_num_threads(4); torch.cuda.set_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    sources = ['flylab/plasticity.py','flylab/resume.py','scripts/extend_plasticity.py','scripts/train_plasticity.py',
               'scripts/train_population.py','flylab/world.py','flylab/controllers.py','flylab/population.py','flylab/population_ppo.py']
    write_json(args.output / 'source-manifest.json', {name: sha(ROOT/name) for name in sources})
    for seed in args.seeds:
        for mode in args.modes:
            run(mode, seed, args)
            torch.cuda.empty_cache()
    if not args.skip_final_evaluations:final_evaluations(args)
    write_json(args.output/'suite-complete.json',{'seeds':args.seeds,'modes':args.modes,'total_steps':args.steps,
               'final_evaluations_complete':not args.skip_final_evaluations})


if __name__ == '__main__': main()
