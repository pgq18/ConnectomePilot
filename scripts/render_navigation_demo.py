"""Render a recorded navigation rollout as PNG frames for an H.264 video."""
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
from PIL import Image,ImageDraw,ImageFont

W,H,S=1600,900,2
FPS=30
BG='#0c1623';PANEL='#142235';FIELD='#101e2e';GRID='#203044'
WHITE='#eef6ff';MUTED='#94a9c0';TEAL='#38d7be';GOLD='#f5bf60'
FONT=None
MONO=None
FONTS={}


def choose_font(explicit, candidates, label):
    if explicit:
        path=Path(explicit).expanduser()
        if not path.is_file():raise ValueError(f'{label} font not found: {path}')
        return str(path)
    for candidate in candidates:
        if Path(candidate).is_file():return candidate
    raise ValueError(f'No {label} font found; pass --font and --mono-font with installed font paths.')


def font(size,mono=False):
    key=(size,mono)
    if key not in FONTS:FONTS[key]=ImageFont.truetype(MONO if mono else FONT,size*S)
    return FONTS[key]


def xy(point):return (40+float(point[0])*90,850-float(point[1])*90)
def box(coords):return tuple(round(float(x)*S) for x in coords)
def points(coords):return [(round(x*S),round(y*S)) for x,y in coords]


def frame(data,activity,t,phase):
    canvas=Image.new('RGB',(W*S,H*S),BG);draw=ImageDraw.Draw(canvas)
    def text(x,y,value,size=22,color=WHITE,mono=False,anchor=None):
        draw.text((round(x*S),round(y*S)),str(value),font=font(size,mono),fill=color,anchor=anchor)
    def line(coords,color,width=1):draw.line(points(coords),fill=color,width=max(1,round(width*S)))
    def rect(bounds,fill,outline=None,width=1,r=0):
        if r:draw.rounded_rectangle(box(bounds),radius=r*S,fill=fill,outline=outline,width=width*S)
        else:draw.rectangle(box(bounds),fill=fill,outline=outline,width=width*S)
    def circle(x,y,r,fill,outline=None,width=1):draw.ellipse(box((x-r,y-r,x+r,y+r)),fill=fill,outline=outline,width=width*S)

    dt=data['control_dt'];count=len(data['commands'])
    position=min(t/dt,count);i=min(int(position),count-1);u=position-i
    if position>=count:i=count-1;u=1.
    a,b=data['states'][i],data['states'][i+1]
    x=a['x']+(b['x']-a['x'])*u;y=a['y']+(b['y']-a['y'])*u
    delta=(b['heading']-a['heading']+math.pi)%(2*math.pi)-math.pi
    heading=a['heading']+delta*u
    state=b if u>=1 else a
    command=data['commands'][i]
    done=position>=count
    status='准备开始' if phase=='intro' else {'success':'到达目标','collision':'发生碰撞','timeout':'回合超时'}.get(state['status'],'正在避障')
    status_color=TEAL if state['status']=='success' else GOLD if state['status']=='running' else '#ff8c94'
    text(40,25,'蝇脑领航 · 避障演示',42)
    text(42,82,'真实连接子图 · 4,096 个神经元 · PPO 419 万步',21,MUTED)
    rect((1150,28,1560,84),PANEL,r=16)
    text(1173,43,'最高实测模型  92.6%',25,TEAL)
    text(1155,96,'训练种子 72 · 单回合仿真回放',17,MUTED)

    rect((40,130,1120,850),FIELD,GRID,2)
    for gx in range(1,12):line([(40+gx*90,130),(40+gx*90,850)],GRID)
    for gy in range(1,8):line([(40,130+gy*90),(1120,130+gy*90)],GRID)
    # A faint start-to-goal guide provides orientation; only the teal line is the driven path.
    for seg in range(0,46,2):
        x1=112+(1039-112)*seg/46;x2=112+(1039-112)*(seg+1)/46
        line([(x1,490),(x2,490)],'#2d3c4c',2)
    for ox,oy,r in data['states'][0]['obstacles']:
        px,py=xy((ox,oy));rr=r*90
        circle(px+3,py+5,rr,'#091321')
        circle(px,py,rr,'#52657b','#8293a6',2)
        circle(px-rr*.22,py-rr*.22,rr*.26,'#60738a')
    goal=data['states'][0]['goal'];gx,gy=xy(goal)
    circle(gx,gy,36,'#1c3435',TEAL if done and state['status']=='success' else '#5b604b',2)
    star=[(gx+math.sin(k*math.pi/5)*(22 if k%2==0 else 10),gy-math.cos(k*math.pi/5)*(22 if k%2==0 else 10)) for k in range(10)]
    draw.polygon(points(star),fill=GOLD)
    text(gx,gy-63,'目标',23,GOLD,anchor='mt')
    sx,sy=xy(data['final_path'][0]);circle(sx,sy,6,MUTED)
    text(sx,sy+31,'起点',19,MUTED,anchor='mt')

    path=[xy(p) for p in data['final_path'][:i+1]]+[xy((x,y))]
    if len(path)>1:line(path,'#134d50',9);line(path,TEAL,4)
    px,py=xy((x,y))
    rays_a=np.asarray(a['observation']['rays']);rays_b=np.asarray(b['observation']['rays'])
    for angle,distance in zip(a['ray_angles'],rays_a+(rays_b-rays_a)*u):
        end=xy((x+distance*math.cos(heading+angle),y+distance*math.sin(heading+angle)))
        line([(px,py),end],'#386878',1)
        circle(*end,2,'#659cac')
    radius=a['radius']*90
    circle(px,py,radius+5,'#184c52')
    circle(px,py,radius,TEAL,WHITE,2)
    nose=(px+math.cos(heading)*(radius+12),py-math.sin(heading)*(radius+12))
    rear=(px-math.cos(heading)*4,py+math.sin(heading)*4)
    side=(-math.sin(heading)*7,-math.cos(heading)*7)
    draw.polygon(points([nose,(rear[0]+side[0],rear[1]+side[1]),(rear[0]-side[0],rear[1]-side[1])]),fill=WHITE)

    rect((1150,130,1560,310),PANEL,r=16)
    text(1172,148,'任务状态',17,MUTED)
    text(1172,176,status,36,status_color)
    text(1172,241,'仿真时间',19,MUTED);text(1535,237,f'{t:4.1f} s',26,mono=True,anchor='rt')
    distance=math.hypot(goal[0]-x,goal[1]-y)
    text(1172,277,'距目标',19,MUTED);text(1535,273,f'{distance:4.2f} m',26,mono=True,anchor='rt')

    rect((1150,330,1560,605),PANEL,r=16)
    text(1172,350,'最后执行的动作' if done else '实时控制',24)
    text(1172,393,'前进速度',19,MUTED)
    text(1535,388,f"{command['final'][0]:.2f} m/s",27,mono=True,anchor='rt')
    colors=['#829bb3',TEAL,WHITE]
    for row,(name,key,color) in enumerate(zip(['基础转向','神经读出修正','最终转向'],['base','residual','final'],colors)):
        yy=443+row*49;value=command[key][1]
        text(1172,yy,name,17,MUTED);text(1535,yy,f'{value:+.2f} rad/s',17,color,mono=True,anchor='rt')
        line([(1172,yy+29),(1537,yy+29)],'#2a3b4f',5)
        center=(1172+1537)/2;end=center+value/2.2*(1537-1172)/2
        line([(center,yy+29),(end,yy+29)],color,5)
        line([(center,yy+24),(center,yy+34)],MUTED)

    rect((1150,625,1560,850),PANEL,r=16)
    text(1172,642,'下降神经元活动',24)
    values=activity[i].reshape(18,73)
    center=np.asarray([20,36,54],dtype=np.float32)
    positive=np.asarray([56,215,190],dtype=np.float32)
    negative=np.asarray([189,129,208],dtype=np.float32)
    signed=values[...,None]
    rgb=center+np.maximum(signed,0)*(positive-center)+np.maximum(-signed,0)*(negative-center)
    heat=Image.fromarray(np.uint8(rgb.clip(0,255))).resize((366*S,110*S),Image.Resampling.NEAREST)
    canvas.paste(heat,(1172*S,685*S))
    text(1172,806,'1,314 个神经元 · 简化模型状态 [-1, 1]',16,MUTED)
    clearance=data['result']['min_clearance'] if done else state['min_clearance']
    text(1172,828,f'截至此刻最小间隙  {clearance*100:.1f} cm',15,TEAL)

    circle(49,878,5,TEAL);text(62,866,'实际行驶轨迹',17,MUTED)
    circle(248,878,5,'#8293a6');text(261,866,'障碍物',17,MUTED)
    circle(386,878,5,GOLD);text(399,866,'目标',17,MUTED)
    text(501,866,'感知线：9 个方向',17,MUTED)
    text(777,866,f"新地图 {data['scene_seed']}",17,MUTED)
    text(1540,866,'回放定格 · 二维仿真' if done else '1× 实时回放 · 二维仿真',17,MUTED,anchor='rt')
    return canvas.resize((W,H),Image.Resampling.LANCZOS)


def main():
    global FONT,MONO
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--font',help='Path to a Chinese-capable TrueType/OpenType font')
    parser.add_argument('--mono-font',help='Path to a monospace TrueType/OpenType font')
    args=parser.parse_args()
    try:
        FONT=choose_font(args.font,[
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'], 'Chinese')
        MONO=choose_font(args.mono_font,[
            '/System/Library/Fonts/SFNSMono.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',FONT], 'monospace')
    except ValueError as error:
        parser.error(str(error))
    FONTS.clear()
    data=json.loads((args.folder/'rollout.json').read_text())
    activity=np.load(args.folder/'neural-activity.npz')['descending']
    assert activity.shape==(len(data['commands']),1314)
    duration=data['result']['steps']*data['control_dt']
    if args.preview:
        frame(data,activity,duration*.55,'motion').save(args.folder/'preview.png')
        frame(data,activity,duration,'end').save(args.folder/'poster.png')
        print(args.folder/'preview.png');return
    folder=args.folder/'frames';folder.mkdir(exist_ok=False)
    schedule=[(0.,'intro')]*FPS+[(min(k/FPS,duration),'motion') for k in range(round(duration*FPS)+1)]+[(duration,'end')]*(3*FPS)
    for idx,(t,phase) in enumerate(schedule):
        frame(data,activity,t,phase).save(folder/f'{idx:05d}.png',compress_level=2)
        if idx%90==0:print(f'{idx}/{len(schedule)} frames',flush=True)
    metadata={'width':W,'height':H,'fps':FPS,'frames':len(schedule),'video_seconds':len(schedule)/FPS,
              'simulation_seconds':duration,'intro_seconds':1,'outro_seconds':3,
              'control_hz':1/data['control_dt'],'playback_speed':1,
              'visual_interpolation':'linear positions; shortest-angle heading between recorded 10 Hz states',
              'model_activity':'signed hidden state of descending neurons in the simplified rate model',
              'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.folder/'render.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
