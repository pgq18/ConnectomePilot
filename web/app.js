const $=id=>document.getElementById(id);
let state=null, scenario='slalom', busy=false, polling=false, commandVersion=0;
const canvas=$('arena'),ctx=canvas.getContext('2d');
const names={base:'仅基础控制器',reflex:'规则避障 · 对照组',fly:'果蝇脑 · 残差控制','ppo-rays':'PPO · 距离传感器','ppo-fly':'PPO · 果蝇脑'};
const hints={base:'只朝目标前进，不使用障碍物信息。',reflex:'用距离传感器直接计算转向，作为工程对照。',fly:'真实连接图参与运算；距离输入与动作输出的映射由人工设定。','ppo-rays':'根据距离与目标信息，由已训练的 PPO 策略修正动作。','ppo-fly':'果蝇连接图固定，PPO 从神经活动与目标信息学习动作修正。'};
async function command(body){if(busy)return;busy=true;commandVersion++;$('error').hidden=true;try{const r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error);state=data;render();}catch(e){$('error').textContent=e.message;$('error').hidden=false;}finally{busy=false;}}
function reset(){return command({command:'reset',mode:$('mode').value,scenario,seed:Number($('seed').value),gain:Number($('gain').value)});}
$('play').onclick=()=>{if(state)command({command:state.paused?'play':'pause'});};
$('reset').onclick=reset;
$('mode').onchange=()=>{ $('mode-help').textContent=hints[$('mode').value];reset();};
document.querySelectorAll('.scenario').forEach(b=>b.onclick=()=>{scenario=b.dataset.value;document.querySelectorAll('.scenario').forEach(x=>x.classList.toggle('active',x===b));reset();});
$('next-seed').onclick=()=>{$('seed').value=Number($('seed').value)+1;reset();};
$('seed').onchange=reset;
$('gain').oninput=()=>{$('gain-label').textContent=Number($('gain').value).toFixed(2)+'×';};
$('gain').onchange=()=>command({command:'gain',value:Number($('gain').value)});
$('default-goal').onclick=()=>{if(state)command({command:'goal',x:state.world.default_goal[0],y:state.world.default_goal[1]});};
function arenaView(){
 const box=canvas.getBoundingClientRect(),w=state.world,pad=23;
 const s=Math.min((box.width-pad*2)/w.width,(box.height-pad*2)/w.height);
 return {box,s,ox:(box.width-w.width*s)/2,oy:(box.height-w.height*s)/2};
}
canvas.addEventListener('click',event=>{
 if(!state||busy)return;
 const {box,s,ox,oy}=arenaView();
 if(s<=0)return;
 const x=(event.clientX-box.left-ox)/s,y=(box.height-oy-(event.clientY-box.top))/s;
 if(x<0||x>state.world.width||y<0||y>state.world.height){
  $('error').textContent='请在地图边框内的空白处选择目标';$('error').hidden=false;return;
 }
 command({command:'goal',x,y});
});
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click();}if(e.key.toLowerCase()==='r')reset();});
$('export').onclick=()=>{if(!state)return;const blob=new Blob([JSON.stringify({exported_at:new Date().toISOString(),...state},null,2)],{type:'application/json'});const u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download=`fly-lab-${state.mode}-${state.world.seed}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};
function metric(id,value,unit){$(id).textContent=value;const small=document.createElement('small');small.textContent=unit;$(id).append(small);}
function render(){if(!state)return;const w=state.world,t=state.telemetry;
 $('connection').textContent='● 已连接';$('connection').classList.add('ok');
 $('arena-mode').textContent=names[state.mode];$('mode').value=state.mode;
 $('mode-help').textContent=hints[state.mode];
 if(state.mode.startsWith('ppo-')&&state.policies?.[state.mode]){const policy=state.policies[state.mode],sceneName={clutter:'散布障碍',slalom:'绕桩',single:'单障碍'}[policy.scenario]||policy.scenario;$('mode-help').textContent+=' 已训练 '+policy.training_steps.toLocaleString()+' 步；训练场景：'+sceneName+'。';}
 const status=w.status==='running'?(state.paused?'就绪 / 已暂停':'实验进行中'):({success:'到达目标',collision:'发生碰撞',timeout:'时间到'})[w.status];
 $('status').textContent=status;$('status').className='status '+w.status;
 $('goal-position').textContent=`${w.goal[0].toFixed(2)}, ${w.goal[1].toFixed(2)} m`;
 $('default-goal').disabled=w.status==='collision'||w.goal.every((v,i)=>v===w.default_goal[i]);
 $('play').innerHTML=state.paused?'继续 / 开始 <span>▶</span>':'暂停实验 <span>Ⅱ</span>';$('play').disabled=w.status!=='running';
 const showOverlay=state.paused&&(w.status!=='running'||(w.steps===0&&w.goal.every((v,i)=>v===w.default_goal[i])));$('overlay').hidden=!showOverlay;
 if(showOverlay){$('overlay').querySelector('span').textContent=w.status==='running'?'READY WHEN YOU ARE':'EPISODE COMPLETE';$('overlay').querySelector('strong').textContent=({success:'已抵达。点击地图设置下一个目标。',collision:'碰撞了。重置场景后可以继续。',timeout:'时间结束。可以换个目标再试。'})[w.status]||(w.steps?'实验已暂停。':'点击地图，选择想去的地方。');$('overlay').querySelector('small').textContent=w.status==='collision'?'重置同一场景会保留所选目标，方便比较控制器。':'在空白处设置目标，点击“继续 / 开始”出发。';}
 metric('distance',w.observation.goal_distance.toFixed(2),'m');metric('travel',w.path_length.toFixed(2),'m');metric('clearance',w.min_clearance.toFixed(2),'m');metric('sim-time',w.sim_seconds.toFixed(1),'s');
 for(const [prefix,key] of [['base','base'],['res','residual']]){ $(prefix+'-v').textContent=t[key][0].toFixed(2);$(prefix+'-w').textContent=t[key][1].toFixed(2);}$('final-w').textContent=t.final[1].toFixed(2);
 const ready=state.brain.available;const flyOption=$('mode').querySelector('option[value=fly]');flyOption.disabled=!ready;flyOption.textContent=ready?'果蝇脑 · 残差控制':'果蝇脑 · 等待连接数据';
 $('brain-title').textContent=ready?'连接数据已就绪':'连接数据尚未就绪';$('brain-title').style.color=ready?'var(--teal)':'var(--amber)';$('brain-message').textContent=ready?'可以选择果蝇脑，开始观察神经网络与小车的闭环。':'先体验规则对照；下载完成后才会启用果蝇脑。';
 for(const mode of ['ppo-rays','ppo-fly']){const option=$('mode').querySelector(`option[value="${mode}"]`),policy=state.policies?.[mode],ready=policy?.available,phase={training:'训练中',evaluating:'评估中',resource_stopped:'资源限制已暂停'}[policy?.status]||'待训练';option.disabled=!ready;option.textContent=names[mode]+(ready?'':`（${phase}）`);}
 $('learning-note').textContent=state.mode==='ppo-fly'?'连接权重固定 · PPO 输出层已训练 · 本轮无学习':'固定连接权重 · 本轮无学习';
 const n=t.neural,neural=['fly','ppo-fly'].includes(state.mode)&&n.neurons;$('neural-placeholder').hidden=!!neural;$('neural-live').hidden=!neural;
 if(neural){$('left-hz').textContent=n.left_hz.toFixed(1)+' Hz';$('right-hz').textContent=n.right_hz.toFixed(1)+' Hz';$('left-bar').style.width=Math.min(100,n.left_hz*2)+'%';$('right-bar').style.width=Math.min(100,n.right_hz*2)+'%';$('neurons').textContent=n.neurons.toLocaleString()+' 神经元';$('compute').textContent=n.compute_ms+' ms / 步';}
 if(state.error){$('error').textContent=state.error;$('error').hidden=false;}
 draw();
}
function draw(){if(!state)return;const w=state.world,{box,s,ox,oy}=arenaView(),dpr=devicePixelRatio||1;
 if(canvas.width!==Math.round(box.width*dpr)||canvas.height!==Math.round(box.height*dpr)){canvas.width=Math.round(box.width*dpr);canvas.height=Math.round(box.height*dpr);}
 ctx.setTransform(dpr,0,0,dpr,0,0);const W=box.width,H=box.height;
 const px=x=>ox+x*s,py=y=>H-oy-y*s;
 ctx.clearRect(0,0,W,H);ctx.fillStyle='#0d171c';ctx.fillRect(0,0,W,H);
 for(let x=0;x<=w.width;x+=.5)for(let y=0;y<=w.height;y+=.5){ctx.fillStyle='#27373b';ctx.beginPath();ctx.arc(px(x),py(y),.7,0,Math.PI*2);ctx.fill();}
 ctx.strokeStyle='#34464a';ctx.lineWidth=1;ctx.strokeRect(ox,oy,w.width*s,w.height*s);
 ctx.font='8px ui-monospace';ctx.fillStyle='#4e686e';ctx.textAlign='center';for(let x=0;x<=w.width;x+=2)ctx.fillText(x+'',px(x),H-oy+14);
 for(const [x,y,r] of w.obstacles){const grd=ctx.createRadialGradient(px(x)-r*s*.3,py(y)-r*s*.4,0,px(x),py(y),r*s);grd.addColorStop(0,'#33434b');grd.addColorStop(1,'#25323a');ctx.fillStyle=grd;ctx.beginPath();ctx.arc(px(x),py(y),r*s,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#48565d';ctx.stroke();ctx.strokeStyle='#36484b';ctx.setLineDash([2,4]);ctx.beginPath();ctx.arc(px(x),py(y),(r+w.radius)*s,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);}
 const [gx,gy]=w.goal;ctx.strokeStyle='#efbe7c44';ctx.lineWidth=1;ctx.beginPath();ctx.arc(px(gx),py(gy),.5*s,0,Math.PI*2);ctx.stroke();ctx.strokeStyle='#efbe7c';ctx.beginPath();ctx.arc(px(gx),py(gy),.25*s,0,Math.PI*2);ctx.stroke();ctx.fillStyle='#efbe7c';ctx.beginPath();ctx.arc(px(gx),py(gy),3,0,Math.PI*2);ctx.fill();ctx.font='9px -apple-system';ctx.fillText('目标',px(gx),py(gy)-.5*s-8);
 ctx.strokeStyle='#8dcab48c';ctx.lineWidth=1.6;ctx.beginPath();w.path.forEach(([x,y],i)=>i?ctx.lineTo(px(x),py(y)):ctx.moveTo(px(x),py(y)));ctx.stroke();
 if($('show-rays').checked){w.observation.rays.forEach((r,i)=>{let a=w.heading+w.ray_angles[i],x=w.x+r*Math.cos(a),y=w.y+r*Math.sin(a);ctx.strokeStyle=r<1?'#e9b07b88':'#65c8a32e';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(px(w.x),py(w.y));ctx.lineTo(px(x),py(y));ctx.stroke();ctx.fillStyle=r<1?'#efbe7c':'#5eaa8e';ctx.beginPath();ctx.arc(px(x),py(y),1.7,0,Math.PI*2);ctx.fill();});}
 ctx.save();ctx.translate(px(w.x),py(w.y));ctx.rotate(-w.heading);const R=Math.max(6,w.radius*s);ctx.shadowColor='#7be5be66';ctx.shadowBlur=16;ctx.fillStyle='#7be5be';ctx.beginPath();ctx.roundRect(-R,-R*.75,R*2,R*1.5,R*.35);ctx.fill();ctx.shadowBlur=0;ctx.fillStyle='#132f25';ctx.beginPath();ctx.moveTo(R*.6,0);ctx.lineTo(-R*.2,-R*.35);ctx.lineTo(-R*.2,R*.35);ctx.closePath();ctx.fill();ctx.fillStyle='#abc9bb';ctx.fillRect(-R*.6,-R,R*1.2,R*.2);ctx.fillRect(-R*.6,R*.8,R*1.2,R*.2);ctx.restore();
 ctx.textAlign='left';ctx.fillStyle='#58756e';ctx.font='8px ui-monospace';ctx.fillText('SEED '+w.seed.toString().padStart(4,'0'),ox+9,oy+16);
}
async function poll(){if(polling)return;polling=true;const version=commandVersion;try{const r=await fetch('/api/state');if(!r.ok)throw Error('服务未响应');const next=await r.json();if(!busy&&version===commandVersion){state=next;render();}}catch(e){if(!busy&&version===commandVersion){$('connection').textContent='连接已断开';$('connection').classList.remove('ok');}}finally{polling=false;setTimeout(poll,150);}}
new ResizeObserver(draw).observe(canvas);poll();
