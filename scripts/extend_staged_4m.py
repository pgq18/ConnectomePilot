"""Extend staged 1M models to 4M; select checkpoints using validation only."""
from pathlib import Path
import argparse
import json
import sys
import time
import subprocess
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flylab.plasticity import PlasticPolicy,PlasticityWorlds,ThreeFactor,GRAPH
from flylab.population_ppo import advantages
from flylab.resume import load_learning_state,save_checkpoint,environment_state
from flylab.continuation import restore_rollout_state
from flylab.selection import select_checkpoints
from scripts.train_population import summarize,write_json
from scripts.extend_plasticity import sha,weights_sha,evaluate


def run(seed, args):
    mode = 'readout'
    out = args.output / f'staged-{seed}'
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    source = args.source / f'staged-{seed}' / 'final.pt'
    checkpoint = torch.load(source, map_location=args.device, weights_only=True)
    base_steps = checkpoint.get('total_steps', checkpoint['config']['steps'])
    assert checkpoint['config']['mode'] == 'readout' and checkpoint['config']['seed'] == seed
    assert checkpoint['config']['experiment'] == 'joint_65536_then_frozen_edges'
    assert 'rollout_state' in checkpoint
    assert base_steps == 1048576 and args.steps > base_steps
    assert checkpoint['config']['graph_sha256'] == sha(GRAPH)
    for name in ['batch','rollout','minibatch']:
        assert getattr(args,name) == checkpoint['config'][name], name
    torch.manual_seed(seed + 2000000); np.random.seed(seed + 2000000)
    model = PlasticPolicy('readout').to(args.device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4) if mode != 'three_factor' else None
    local = ThreeFactor(model, args.batch) if mode == 'three_factor' else None
    load_learning_state(checkpoint, model, optimizer, local, restore_traces='rollout_state' in checkpoint)
    frozen_edges = model.edge_weight.detach().clone()
    initial_digest = weights_sha(model)
    initial_optimizer_steps = sorted({int(s['step']) for s in optimizer.state.values()})
    assert initial_optimizer_steps == [65536]
    assert {n for n,p in model.named_parameters() if p.requires_grad} == {'actor.weight','actor.bias','critic.weight','critic.bias'}
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(mode=mode, seed=seed, output=str(out), base_steps=base_steps,
        experiment='joint_65536_then_frozen_edges', source_mode='readout', frozen_since_steps=65536,
        initial_optimizer_steps=initial_optimizer_steps, frozen_parameters=['edge_weight'],
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
    raw,hidden,starts = restore_rollout_state(checkpoint,env,args.device)
    assert environment_state(env) == checkpoint['rollout_state']['environment']
    assert np.array_equal(raw, checkpoint['rollout_state']['raw'].cpu().numpy())
    assert torch.equal(hidden, checkpoint['rollout_state']['hidden'])
    assert torch.equal(starts, checkpoint['rollout_state']['starts'])
    assert torch.equal(torch.get_rng_state(),checkpoint['rollout_state']['torch_cpu_rng'].cpu())
    assert torch.equal(torch.cuda.get_rng_state(args.device),checkpoint['rollout_state']['torch_device_rng'].cpu())
    write_json(out/'resume-audit.json',{'full_environment_restored':True,'observations_and_neural_state_restored':True,
        'cpu_and_gpu_rng_restored':True,'source_checkpoint_sha256':sha(source),
        'model_state_sha256':weights_sha(model),'optimizer_steps':initial_optimizer_steps})
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
        if not torch.equal(model.edge_weight, frozen_edges): raise AssertionError('Frozen edges changed')
        if model.edge_weight.grad is not None: raise AssertionError('Frozen edges acquired a gradient')
        recent = [row for row in episodes if row['environment_steps'] > completed-args.batch*args.rollout]
        training_seconds += time.perf_counter()-started
        row = {'mode': mode, 'seed': seed, 'steps': completed, 'seconds': training_seconds,
               'gpu_peak_mib': torch.cuda.max_memory_allocated()/2**20,
               'training_episodes': len(episodes), 'recent': summarize(recent) if recent else None, **update_stats}
        logs.append(row); write_json(out / 'progress.json', row)
        if completed % 65536 == 0 or completed == args.steps: print(json.dumps(row), flush=True)
        if completed in args.milestones or completed == args.steps:
            filename='final.pt' if completed==args.steps else f'checkpoint-{completed}.pt'
            save_checkpoint(out/filename,model,optimizer,local,config,completed,env,raw,hidden,starts)
            score=evaluate(model,validation)
            write_json(out/f'validation-{completed}.json',score)
            print(json.dumps({'validation':mode,'seed':seed,'steps':completed,**score['summary']}),flush=True)
    train_seconds = training_seconds
    write_json(out / 'training.json', {'updates': logs, 'episodes': episodes})
    measurements = {'frozen_edges_equal_source': torch.equal(model.edge_weight, frozen_edges),
        'edge_change_since_freeze_max': float((model.edge_weight-frozen_edges).abs().max()),
        'final_optimizer_steps': sorted({int(s['step']) for s in optimizer.state.values()}),
        'train_seconds': train_seconds, 'gpu_peak_mib': torch.cuda.max_memory_allocated()/2**20,
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


def evaluation_paths(args,seed):
    paths={1048576:args.source/f'staged-{seed}/final.pt'}
    for step in sorted(set([*args.milestones,args.steps])):
        if 1048576<step<=args.steps:
            name='final.pt' if step==args.steps else f'checkpoint-{step}.pt'
            paths[step]=args.output/f'staged-{seed}'/name
    return paths


def final_evaluations(args):
    # Commit validation-based choices to disk BEFORE opening any fresh test map.
    candidates=[]
    for seed in args.seeds:
        for step,path in evaluation_paths(args,seed).items():
            validation_path=args.output/f'staged-{seed}/validation-{step}.json'
            score=json.loads(validation_path.read_text())
            scenes=sorted(row['scene_seed'] for row in score['episodes'])
            assert scenes==list(range(args.validation_start,args.validation_start+args.validation_episodes))
            candidates.append({'seed':seed,'total_steps':step,
                'successes':score['summary']['successes'],'episodes':score['summary']['episodes'],
                'checkpoint':str(path),'checkpoint_sha256':sha(path),
                'validation_sha256':sha(validation_path)})
    selection=select_checkpoints(candidates)
    selection.update(validation_start=args.validation_start,validation_episodes=args.validation_episodes,
        fresh_test_start=args.test_start,fresh_test_episodes=args.test_episodes,
        chosen_before_test=True,selected_at_unix=time.time())
    write_json(args.output/'selection.json',selection)
    print(json.dumps({'selection':selection}),flush=True)
    fresh=list(range(args.test_start,args.test_start+args.test_episodes))
    previous=list(range(5700000,5700200))
    folder=args.output/'evaluation';folder.mkdir()
    write_json(folder/'test-start.json',{'started_at_unix':time.time(),'selection_sha256':sha(args.output/'selection.json')})
    for seed in args.seeds:
        selected_step=selection['per_seed'][str(seed)]['total_steps']
        for step,path in evaluation_paths(args,seed).items():
            cp=torch.load(path,map_location=args.device,weights_only=True)
            model=PlasticPolicy('readout').to(args.device);model.load_state_dict(cp['model'],strict=True)
            score=evaluate(model,fresh)
            score.update(total_steps=step,seed=seed,checkpoint=str(path),checkpoint_sha256=sha(path))
            write_json(folder/f'step-{step}-{seed}.json',score)
            print(json.dumps({'test_steps':step,'seed':seed,**score['summary']}),flush=True)
            if step in {1048576,selected_step,args.steps}:
                write_json(args.output/f'staged-{seed}/previous-test-{step}.json',evaluate(model,previous))
            del model,cp
        torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'results/plasticity/staged-long/main')
    p.add_argument('--seeds',type=int,nargs='+',default=[71,72,73])
    p.add_argument('--steps',type=int,default=4194304)
    p.add_argument('--milestones',type=int,nargs='+',default=[1572864,2097152,2621440,3145728,3670016,4194304])
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--rollout',type=int,default=128)
    p.add_argument('--minibatch',type=int,default=64)
    p.add_argument('--validation-start',type=int,default=5410000)
    p.add_argument('--validation-episodes',type=int,default=200)
    p.add_argument('--test-start',type=int,default=5800000)
    p.add_argument('--test-episodes',type=int,default=500)
    p.add_argument('--skip-final-evaluations',action='store_true')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--parallel-runs',action='store_true')
    p.add_argument('--worker-seed',type=int,help=argparse.SUPPRESS)
    args=p.parse_args()
    if not torch.cuda.is_available():p.error('This runner requires the workstation GPU.')
    if args.steps<=1048576 or args.steps%(args.batch*args.rollout):p.error('Use a larger whole-rollout budget.')
    if any(x%(args.batch*args.rollout) or x<=1048576 or x>args.steps for x in args.milestones):p.error('Invalid milestone.')
    if not 5410000<=args.validation_start<args.validation_start+args.validation_episodes<=5500000 or not 5800000<=args.test_start<args.test_start+args.test_episodes<=5900000:
        p.error('Use reserved validation and fresh-test ranges.')
    if args.worker_seed is not None:
        torch.set_num_threads(4);torch.cuda.set_device(args.device)
        run(args.worker_seed,args)
        return
    if args.output.exists():p.error('Use a new output directory.')
    torch.set_num_threads(4);torch.cuda.set_device(args.device);args.output.mkdir(parents=True)
    sources=['flylab/plasticity.py','flylab/resume.py','flylab/continuation.py','scripts/extend_staged.py','scripts/extend_staged_4m.py','flylab/selection.py',
        'docs/STAGED_4M_PROTOCOL.md',
        'scripts/train_frozen_readout.py','scripts/extend_plasticity.py','scripts/train_plasticity.py',
        'scripts/train_population.py','flylab/world.py','flylab/controllers.py','flylab/population.py','flylab/population_ppo.py']
    write_json(args.output/'source-manifest.json',{name:sha(ROOT/name) for name in sources})
    inputs={str((args.source/f'staged-{seed}/final.pt').resolve().relative_to(ROOT)):
            sha(args.source/f'staged-{seed}/final.pt') for seed in args.seeds}
    write_json(args.output/'input-manifest.json',inputs)
    if args.parallel_runs:
        processes=[]
        try:
            for seed in args.seeds:
                log=(args.output/f'worker-{seed}.log').open('w')
                process=subprocess.Popen([sys.executable,__file__,*sys.argv[1:],'--worker-seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
                processes.append((seed,process,log))
            failures=[(seed,process.wait()) for seed,process,log in processes]
            if any(code for seed,code in failures):raise RuntimeError(f'Worker failures: {failures}')
        finally:
            for seed,process,log in processes:
                if process.poll() is None:process.terminate()
                log.close()
    else:
        for seed in args.seeds:
            run(seed,args);torch.cuda.empty_cache()
    if not args.skip_final_evaluations:final_evaluations(args)
    write_json(args.output/'suite-complete.json',{'seeds':args.seeds,'total_steps':args.steps,
        'milestones':sorted(set([1048576,*args.milestones,args.steps])),
        'method':'joint_65536_then_frozen_edges','final_evaluations_complete':not args.skip_final_evaluations})


if __name__=='__main__':main()
