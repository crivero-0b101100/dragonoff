"""Headless VST3 instrument backend. All notes use absolute quarter-note positions.

Render one bar at a time with persistent plugin state, explicit note-offs and
bar automation. No GUI, device, transport clock or live DAW is required.
"""
from pathlib import Path
import json,math
import numpy as np
import soundfile as sf
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
from pedalboard import load_plugin

def expand_events(notes,bpm):
    spb=60/bpm; events=[]
    for n in notes:
        t=float(n['onset_beats'])*spb; off=t+float(n['duration_beats'])*spb
        events.append((t,1,int(n['pitch']),int(n.get('velocity',90))))
        events.append((off,0,int(n['pitch']),0))
    return sorted(events,key=lambda e:(e[0],e[1]))

def interpolate(points,bar,default):
    if not points:return default
    points=sorted(points)
    if bar<=points[0][0]:return points[0][1]
    for (a,x),(b,y) in zip(points,points[1:]):
        if a<=bar<=b:return x+(y-x)*(bar-a)/(b-a)
    return points[-1][1]

def render_vst(track,session,path,plugin_path):
    sr=session['sample_rate'];bpm=session['bpm'];bars=session['bars']; spbar=240/bpm
    framesbar=round(sr*spbar); p=load_plugin(str(plugin_path))
    for key,value in track['patch'].items():
        if key not in p.parameters:raise ValueError(f"Unknown plugin parameter {key}")
        if not 0<=value<=1:raise ValueError(f"Raw parameter {key} outside 0..1")
        p.parameters[key].raw_value=value
    events=expand_events(track['notes'],bpm); idx=0
    with sf.SoundFile(str(path),'w',sr,channels=2,subtype='FLOAT') as f:
        for bar0 in range(bars):
            start=bar0*spbar;stop=(bar0+1)*spbar; messages=[]
            while idx<len(events) and events[idx][0]<stop-1e-9:
                t,kind,pitch,velocity=events[idx]
                if t>=start-1e-9:
                    messages.append(Message('note_on' if kind else 'note_off',note=pitch,velocity=velocity,time=max(0,t-start)))
                idx+=1
            for param,points in track.get('automation',{}).items():
                p.parameters[param].raw_value=float(interpolate(points,bar0+1,track['patch'].get(param,.5)))
            # VST hosts may floor duration * sample_rate. Request the exact
            # integer-grid duration so tempos with fractional samples per bar
            # (for example 126 BPM at 48 kHz) cannot return a frame-short block.
            audio=p(messages,duration=framesbar/sr,sample_rate=sr,num_channels=2,buffer_size=2048,reset=(bar0==0))
            if audio.shape[1]!=framesbar:raise RuntimeError('Plugin returned a non-grid block')
            if not np.isfinite(audio).all():raise RuntimeError('Plugin generated non-finite audio')
            f.write(audio.T)
    probe=track.get('voicing_probe')
    if probe:
        audio,rate=sf.read(str(path),start=round(probe['start_seconds']*sr),frames=round(probe.get('duration_seconds',1)*sr),always_2d=True,dtype='float32')
        mono=audio.mean(axis=1); spectrum=np.abs(np.fft.rfft(mono*np.hanning(len(mono))))
        frequencies=np.fft.rfftfreq(len(mono),1/rate);levels=[]
        for pitch in probe['pitches']:
            hz=440*2**((pitch-69)/12)
            band=spectrum[np.abs(frequencies-hz)<max(3,hz*.014)]
            levels.append(float(np.max(band)))
        if min(levels)<max(levels)*10**(probe.get('minimum_relative_db',-35)/20):
            raise RuntimeError(f"{track['id']}: a written chord voice is missing in rendered audio; inspect plugin polyphony. Levels: {levels}")
    return path

def write_midi(session,path):
    mf=MidiFile(type=1,ticks_per_beat=960);meta=MidiTrack();mf.tracks.append(meta)
    key = session['key'] + ('m' if session.get('scale', '').endswith('minor') else '')
    meta.extend([MetaMessage('track_name',name=session['title'].encode('ascii','replace').decode('ascii'),time=0),MetaMessage('set_tempo',tempo=bpm2tempo(session['bpm']),time=0),MetaMessage('time_signature',numerator=4,denominator=4,time=0),MetaMessage('key_signature',key=key,time=0)])
    last_marker=0
    for sec in session.get('sections',[]):
        tick=round((sec['start_bar']-1)*4*960)
        meta.append(MetaMessage('marker',text=sec['name'].encode('ascii','replace').decode('ascii'),time=tick-last_marker))
        last_marker=tick
    meta.append(MetaMessage('end_of_track',time=session['bars']*4*960-last_marker))
    for ti,t in enumerate(session['tracks']):
        if not t.get('notes'):continue
        mt=MidiTrack();mf.tracks.append(mt);mt.append(MetaMessage('track_name',name=t['name'].encode('ascii','replace').decode('ascii'),time=0))
        ev=[]
        for n in t['notes']:
            on=round(n['onset_beats']*960);off=round((n['onset_beats']+n['duration_beats'])*960)
            ev.append((on,1,n['pitch'],n['velocity']));ev.append((off,0,n['pitch'],0))
        last=0
        for tick,on,pitch,vel in sorted(ev,key=lambda e:(e[0],e[1])):
            mt.append(Message('note_on' if on else 'note_off',channel=(ti%8),note=pitch,velocity=vel,time=tick-last));last=tick
    mf.save(str(path))

if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('session',type=Path)
    ap.add_argument('output_directory',type=Path)
    ap.add_argument('--plugin',type=Path,required=True)
    ap.add_argument('--track',default=None)
    args=ap.parse_args()
    session=json.loads(args.session.read_text())
    args.output_directory.mkdir(parents=True,exist_ok=True)
    write_midi(session,args.output_directory/'composition.mid')
    for track in session['tracks']:
        if track.get('source')!='vst3' or args.track and args.track!=track['id']:continue
        path=args.output_directory/(track['id']+'.wav')
        render_vst(track,session,path,args.plugin)
        print(str(path),flush=True)
