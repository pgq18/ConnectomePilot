"""Encode only a complete frame sequence, validate it, then publish the MP4."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import shutil


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('--ffmpeg',default='ffmpeg',help='FFmpeg executable path or command name')
    parser.add_argument('--ffprobe',default='ffprobe',help='FFprobe executable path or command name')
    args=parser.parse_args();folder=args.folder
    ffmpeg=shutil.which(args.ffmpeg);ffprobe=shutil.which(args.ffprobe)
    if not ffmpeg or not ffprobe:
        parser.error('FFmpeg and FFprobe are required. Install them in the project Conda environment or pass executable paths.')
    render=json.loads((folder/'render.json').read_text())
    expected=[f'{i:05d}.png' for i in range(render['frames'])]
    assert sorted(p.name for p in (folder/'frames').glob('*.png'))==expected
    temporary=folder/'avoidance.pending.mp4'
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-xerror','-n',
        '-framerate',str(render['fps']),'-i',str(folder/'frames/%05d.png'),
        '-frames:v',str(render['frames']),'-c:v','libx264','-preset','medium','-crf','18',
        '-pix_fmt','yuv420p','-movflags','+faststart','-threads','4',str(temporary)],check=True)
    info=json.loads(subprocess.check_output([ffprobe,'-v','error','-show_entries',
        'stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration,size',
        '-of','json',str(temporary)],text=True))
    stream=info['streams'][0]
    assert int(stream['nb_frames'])==render['frames'],stream
    assert (stream['width'],stream['height'])==(render['width'],render['height'])
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
    assert stream['avg_frame_rate']==f"{render['fps']}/1"
    assert abs(float(stream['duration'])-render['frames']/render['fps'])<.001
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-xerror','-i',str(temporary),'-f','null','-'],check=True)
    temporary.replace(folder/'avoidance.mp4')
    (folder/'video-info.json').write_text(json.dumps(info,indent=2))
    for name,seconds in [('qa-middle.png',6),('qa-end.png',render['video_seconds']-.5)]:
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-ss',str(seconds),
            '-i',str(folder/'avoidance.mp4'),'-frames:v','1',str(folder/name)],check=True)
    result={'verified_frames':render['frames'],'video_sha256':hashlib.sha256((folder/'avoidance.mp4').read_bytes()).hexdigest(),
            'complete_decode_passed':True,'encoder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder/'encoding-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
