// src/components/AccessibilityPanel.tsx
import { Eye, EyeOff, Palette, Volume2, VolumeX, Hand, MousePointer2, Type } from 'lucide-react';

export type VisualLevel = 'none' | 'mild' | 'moderate' | 'severe' | 'central' | 'peripheral';
export type ColorBlind = 'none' | 'protanopia' | 'deuteranopia' | 'tritanopia' | 'achroma';
export type HearingMode = 'normal' | 'mute' | 'lowpass' | 'highpass';
export type MotorMode = 'normal' | 'singlehand';
export type CognitiveMode = 'normal' | 'dyslexia' | 'simplify';

export default function AccessibilityPanel(props: {
  visual: VisualLevel;
  setVisual: (v: VisualLevel) => void;
  color: ColorBlind;
  setColor: (c: ColorBlind) => void;
  hearing: HearingMode;
  setHearing: (h: HearingMode) => void;
  motor: MotorMode;
  setMotor: (m: MotorMode) => void;
  cognitive: CognitiveMode;
  setCognitive: (c: CognitiveMode) => void;
}) {
  const Item = ({active, onClick, children}:{active:boolean;onClick:()=>void;children:React.ReactNode}) => (
    <button
      onClick={onClick}
      className={`px-3 py-2 rounded-lg text-sm border transition ${active ? 'bg-blue-600 text-white border-blue-500' : 'bg-gray-700/40 border-gray-600 hover:bg-gray-700'}`}
    >{children}</button>
  );

  return (
    <div className="bg-gray-800/90 backdrop-blur rounded-xl p-4 shadow-xl space-y-4">
      {/* Visual */}
      <div>
        <div className="flex items-center gap-2 mb-2"><Eye className="w-4 h-4 text-yellow-400"/><b>Visual</b></div>
        <div className="flex flex-wrap gap-2">
          {(['none','mild','moderate','severe','central','peripheral'] as VisualLevel[]).map(v=>(
            <Item key={v} active={props.visual===v} onClick={()=>props.setVisual(v)}>{v}</Item>
          ))}
        </div>
      </div>

      {/* Color */}
      <div>
        <div className="flex items-center gap-2 mb-2"><Palette className="w-4 h-4 text-purple-400"/><b>Color Blindness</b></div>
        <div className="flex flex-wrap gap-2">
          {(['none','protanopia','deuteranopia','tritanopia','achroma'] as ColorBlind[]).map(c=>(
            <Item key={c} active={props.color===c} onClick={()=>props.setColor(c)}>{c}</Item>
          ))}
        </div>
      </div>

      {/* Hearing */}
      <div>
        <div className="flex items-center gap-2 mb-2"><Volume2 className="w-4 h-4 text-blue-400"/><b>Hearing</b></div>
        <div className="flex flex-wrap gap-2">
          {(['normal','mute','lowpass','highpass'] as HearingMode[]).map(h=>(
            <Item key={h} active={props.hearing===h} onClick={()=>props.setHearing(h)}>{h}</Item>
          ))}
        </div>
      </div>

      {/* Motor */}
      <div>
        <div className="flex items-center gap-2 mb-2"><Hand className="w-4 h-4 text-emerald-400"/><b>Motor</b></div>
        <div className="flex flex-wrap gap-2">
          {(['normal','singlehand'] as MotorMode[]).map(m=>(
            <Item key={m} active={props.motor===m} onClick={()=>props.setMotor(m)}>{m}</Item>
          ))}
        </div>
      </div>

      {/* Cognitive */}
      <div>
        <div className="flex items-center gap-2 mb-2"><Type className="w-4 h-4 text-pink-400"/><b>Cognitive</b></div>
        <div className="flex flex-wrap gap-2">
          {(['normal','dyslexia','simplify'] as CognitiveMode[]).map(c=>(
            <Item key={c} active={props.cognitive===c} onClick={()=>props.setCognitive(c)}>{c}</Item>
          ))}
        </div>
      </div>
    </div>
  );
}
