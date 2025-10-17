// src/hooks/useAudioFilter.ts
import { useEffect, useRef } from 'react';

export type FilterType = 'none'|'lowpass'|'highpass';

export default function useAudioFilter(audioEl: HTMLAudioElement | null, type: FilterType){
  const ctxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const biquadRef = useRef<BiquadFilterNode | null>(null);

  useEffect(()=>{
    if(!audioEl) return;
    if(!ctxRef.current) ctxRef.current = new AudioContext();
    const ctx = ctxRef.current;

    if(sourceRef.current) sourceRef.current.disconnect();
    if(biquadRef.current) biquadRef.current.disconnect();

    const source = ctx.createMediaElementSource(audioEl);
    const biquad = ctx.createBiquadFilter();

    if(type==='lowpass'){ biquad.type = 'lowpass'; biquad.frequency.value = 1200; }
    else if(type==='highpass'){ biquad.type = 'highpass'; biquad.frequency.value = 1200; }
    else { biquad.type = 'allpass'; }

    source.connect(biquad).connect(ctx.destination);

    sourceRef.current = source;
    biquadRef.current = biquad;

    return ()=>{ source.disconnect(); biquad.disconnect(); };
  },[audioEl, type]);
}
