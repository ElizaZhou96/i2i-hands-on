import React, { useState, useRef, useCallback, useEffect, CSSProperties } from 'react';
import Webcam from 'react-webcam';
import { Camera, Scan, Settings, ChevronDown } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import ColorBlindSVGDefs from './components/ColorBlindSVGDefs';

declare global {
  interface Window {
    webkitSpeechRecognition?: any;
    SpeechRecognition?: any;
  }
}

type VisualLevel = 'none' | 'mild' | 'moderate' | 'severe' | 'central' | 'peripheral';
type ColorBlind = 'none' | 'protanopia' | 'deuteranopia' | 'tritanopia' | 'achroma';
type HearingMode = 'normal' | 'mute';
type MotorMode = 'normal' | 'singlehand';
type CognitiveMode = 'normal' | 'dyslexia' | 'simplify';

function App() {
  const [isDetectionActive, setIsDetectionActive] = useState(false);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [detectedObjects, setDetectedObjects] = useState<Array<{ class: string; score: number }>>([]);
  const previousObjectsRef = useRef<Set<string>>(new Set());
  const spokenOnceRef = useRef<Set<string>>(new Set());
  const webcamRef = useRef<Webcam | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number>();
  const isDetectionRunningRef = useRef(false);
  const [isCameraReady, setIsCameraReady] = useState(false);

  const [visual, setVisual] = useState<VisualLevel>('none');
  const [colorBlind, setColorBlind] = useState<ColorBlind>('none');
  const [hearing, setHearing] = useState<HearingMode>('normal');
  const [motor, setMotor] = useState<MotorMode>('normal');
  const [cognitive, setCognitive] = useState<CognitiveMode>('normal');

  const [controlsOpen, setControlsOpen] = useState(false);

  const [detCaptions, setDetCaptions] = useState<string[]>([]);
  const [hearingCaptions, setHearingCaptions] = useState<string[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const recognitionRef = useRef<any>(null);
  const asrRestartRef = useRef(false);
  const lastAsrTextRef = useRef<string>('');
  const asrDebounceRef = useRef<number | null>(null);

  useEffect(() => {
    if (!audioRef.current) return;
    audioRef.current.muted = hearing !== 'normal';
  }, [hearing]);

  const startASR = useCallback(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;
    if (recognitionRef.current) return;
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = navigator.language || 'en-US';
    rec.onresult = (e: any) => {
      let finalText = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) {
          finalText += e.results[i][0].transcript;
        }
      }
      const clean = finalText.trim();
      if (!clean) return;
      if (clean === lastAsrTextRef.current) return;
      lastAsrTextRef.current = clean;
      if (asrDebounceRef.current) window.clearTimeout(asrDebounceRef.current);
      asrDebounceRef.current = window.setTimeout(() => {
        setHearingCaptions(prev => [clean, ...prev].slice(0, 6));
      }, 120);
    };
    rec.onend = () => {
      recognitionRef.current = null;
      if (asrRestartRef.current) startASR();
    };
    recognitionRef.current = rec;
    asrRestartRef.current = true;
    rec.start();
  }, []);

  useEffect(() => {
    if (hearing === 'normal') {
      asrRestartRef.current = false;
      recognitionRef.current?.stop();
      recognitionRef.current = null;
      setHearingCaptions([]);
      lastAsrTextRef.current = '';
      if (asrDebounceRef.current) window.clearTimeout(asrDebounceRef.current);
      asrDebounceRef.current = null;
      return;
    }
    startASR();
    return () => {
      asrRestartRef.current = false;
      recognitionRef.current?.stop();
      recognitionRef.current = null;
    };
  }, [hearing, startASR]);

  const speak = useCallback((text: string) => {
    if (hearing !== 'mute') {
      speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = navigator.language || 'en-US';
      speechSynthesis.speak(u);
    }
    setDetCaptions(prev => [text, ...prev].slice(0, 6));
  }, [hearing]);

  useEffect(() => {
    return () => {
      stopDetection();
    };
  }, []);

  useEffect(() => {
    const initTF = async () => {
      try {
        await tf.ready();
        await tf.setBackend('webgl');
        const loadedModel = await cocoSsd.load({ base: 'mobilenet_v2' });
        setModel(loadedModel);
      } catch (error) {
        console.error('TF init/model error:', error);
      }
    };
    initTF();
  }, []);

  const colorMap: { [key: string]: string } = {
    person: '#0072B2',
    book: '#56B4E9',
    'cell phone': '#009E73',
    laptop: '#E69F00',
    tv: '#CC79A7',
    bottle: '#F0E442',
    chair: '#8A4FAA',
    cup: '#5AC8FA',
    keyboard: '#D55E00',
    mouse: '#8E8E93',
    remote: '#9C27B0',
    backpack: '#795548',
    default: '#8E8E93'
  };

  const indoorObjects = [
    'person', 'book', 'cell phone', 'laptop', 'tv', 'bottle',
    'chair', 'cup', 'keyboard', 'mouse', 'remote', 'backpack'
  ];

  const getColorForClass = (className: string) => colorMap[className] || colorMap.default;

  const stopDetection = useCallback(() => {
    isDetectionRunningRef.current = false;
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = undefined;
    }
    speechSynthesis.cancel();
    setDetectedObjects([]);
    previousObjectsRef.current.clear();
    spokenOnceRef.current.clear();
    setDetCaptions([]);
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx && canvasRef.current) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  }, []);

  const detectObjects = useCallback(async () => {
    if (!model || !webcamRef.current || !canvasRef.current || !isDetectionRunningRef.current) return;

    const video = webcamRef.current.video as HTMLVideoElement | null;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!video || !ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    try {
      const predictions = await model.detect(video, undefined, 0.6);
      if (!isDetectionRunningRef.current) return;

      const present = new Set(predictions.map(p => p.class));
      const newOnes: string[] = [];
      present.forEach(c => {
        if (!previousObjectsRef.current.has(c)) newOnes.push(c);
      });

      setDetectedObjects(predictions.map(p => ({ class: p.class, score: p.score })));

      newOnes.forEach(c => {
        spokenOnceRef.current.add(c);
        speak(`Detected ${c}`);
      });

      const gone: string[] = [];
      previousObjectsRef.current.forEach(c => {
        if (!present.has(c)) gone.push(c);
      });
      gone.forEach(c => spokenOnceRef.current.delete(c));

      previousObjectsRef.current = present;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      predictions.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        const color = getColorForClass(prediction.class);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        ctx.font = 'bold 16px Inter, system-ui, sans-serif';
        const label = `${prediction.class} ${(prediction.score * 100).toFixed(1)}%`;
        const metrics = ctx.measureText(label);
        const pad = 8;
        ctx.fillStyle = color;
        ctx.fillRect(x - pad / 2, y - 30, metrics.width + pad * 2, 30);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, x + pad / 2, y - 10);
      });

      if (isDetectionRunningRef.current) {
        animationFrameRef.current = requestAnimationFrame(detectObjects);
      }
    } catch (error) {
      console.error('Detection error:', error);
      setIsDetectionActive(false);
      stopDetection();
    }
  }, [model, speak, stopDetection]);

  useEffect(() => {
    if (isDetectionActive) {
      isDetectionRunningRef.current = true;
      detectObjects();
    } else {
      stopDetection();
    }
  }, [isDetectionActive, detectObjects, stopDetection]);

  const handleUserMedia = useCallback(() => {
    setIsCameraReady(true);
  }, []);

  const visualClass = (() => {
    switch (visual) {
      case 'mild': return 'vis-mild';
      case 'moderate': return 'vis-moderate';
      case 'severe': return 'vis-severe';
      case 'central': return 'vis-central';
      case 'peripheral': return 'vis-peripheral';
      default: return '';
    }
  })();

  const colorFilterId = (() => {
    switch (colorBlind) {
      case 'protanopia': return '#cb-protanopia';
      case 'deuteranopia': return '#cb-deuteranopia';
      case 'tritanopia': return '#cb-tritanopia';
      case 'achroma': return '#cb-achroma';
      default: return '';
    }
  })();

  const colorFilterStyle: CSSProperties = colorFilterId
    ? { filter: `url(${colorFilterId})`, WebkitFilter: `url(${colorFilterId})` }
    : {};

  const longPress = (fn: () => void, ms = 600) => {
    let t: number | undefined;
    return {
      onMouseDown: (e: React.MouseEvent) => {
        if (motor !== 'singlehand') { fn(); return; }
        const el = (e.currentTarget as HTMLElement);
        el.dataset.press = 'on';
        t = window.setTimeout(() => { fn(); el.dataset.press = ''; }, ms);
      },
      onMouseUp: (e: React.MouseEvent) => { if (t) { clearTimeout(t); t = undefined; (e.currentTarget as HTMLElement).dataset.press = ''; } },
      onMouseLeave: (e: React.MouseEvent) => { if (t) { clearTimeout(t); t = undefined; (e.currentTarget as HTMLElement).dataset.press = ''; } },
    };
  };

  const originalTextMapRef = useRef<WeakMap<Text, string>>(new WeakMap());

  useEffect(() => {
    const root = document.querySelector('.cognitive-scope');
    if (!root) return;

    if (cognitive === 'dyslexia') {
      function scrambleWord(w: string) {
        if (w.length <= 3) return w;
        const mid = w.slice(1, -1).split('');
        for (let i = mid.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [mid[i], mid[j]] = [mid[j], mid[i]];
        }
        return w[0] + mid.join('') + w[w.length - 1];
      }
      function scrambleTextNode(node: Text) {
        const orig = node.textContent ?? '';
        if (!originalTextMapRef.current.get(node)) {
          originalTextMapRef.current.set(node, orig);
        }
        const scrambled = orig.replace(/[A-Za-zÀ-ÿ]{4,}/g, scrambleWord);
        if (scrambled !== orig) node.textContent = scrambled;
      }
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
        acceptNode(n) {
          const p = (n as Text).parentElement;
          if (!p) return NodeFilter.FILTER_REJECT;
          if (p.closest('button, .no-scramble, code, pre, input, textarea, svg')) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        }
      } as unknown as NodeFilter);
      const targets: Text[] = [];
      while (walker.nextNode()) targets.push(walker.currentNode as Text);
      targets.forEach(scrambleTextNode);
    } else {
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      const toRestore: Text[] = [];
      while (walker.nextNode()) {
        const t = walker.currentNode as Text;
        const orig = originalTextMapRef.current.get(t);
        if (orig !== undefined && t.textContent !== orig) toRestore.push(t);
      }
      toRestore.forEach((t) => {
        const orig = originalTextMapRef.current.get(t);
        if (orig !== undefined) t.textContent = orig;
      });
    }
  }, [cognitive]);

  return (
    <>
      <ColorBlindSVGDefs />
      <div className={`min-h-screen bg-gray-900 text-white ${motor === 'singlehand' ? 'motor-singlehand' : ''} ${cognitive === 'simplify' ? 'cog-simplify' : ''}`} style={colorFilterStyle}>
        <header className="fixed top-0 left-0 right-0 z-50">
          <div className="bg-gray-900/85 backdrop-blur border-b border-white/10">
            <div className="max-w-7xl mx-auto px-4 h-12 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Camera className="w-5 h-5" />
                <span className="font-semibold">Accessibility Simulator</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  {...longPress(() => setIsDetectionActive(!isDetectionActive))}
                  className={`dwell px-3 py-1.5 rounded-md text-sm font-medium transition ${isDetectionActive ? 'bg-blue-600 hover:bg-blue-700' : 'bg-gray-700 hover:bg-gray-600'}`}
                  disabled={!model}
                >
                  <div className="flex items-center gap-2">
                    <Scan className="w-4 h-4" />
                    <span>{isDetectionActive ? 'Stop' : 'Start'}</span>
                  </div>
                </button>
                <button
                  onClick={() => setControlsOpen(v => !v)}
                  className="px-3 py-1.5 rounded-md text-sm font-medium bg-gray-700 hover:bg-gray-600 flex items-center gap-2"
                >
                  <Settings className="w-4 h-4" />
                  <span>Modes</span>
                  <ChevronDown className={`w-4 h-4 transition ${controlsOpen ? 'rotate-180' : ''}`} />
                </button>
              </div>
            </div>
          </div>
        </header>

        <main className="container mx-auto px-4 pt-16 pb-6">
          {controlsOpen && (
            <div className="mb-4 bg-gray-800/95 rounded-xl p-4 space-y-4">
              <div className="grid md:grid-cols-2 gap-3">
                <div className="flex flex-wrap gap-2 items-center">
                  <span className="text-sm text-gray-300 w-20">Visual</span>
                  {(['none','mild','moderate','severe','central','peripheral'] as VisualLevel[]).map(v=>(
                    <button key={v} onClick={()=>setVisual(v)} className={`px-3 py-1.5 rounded text-sm border ${visual===v?'bg-blue-600 text-white border-blue-500':'bg-gray-700/50 border-gray-600 hover:bg-gray-700'}`}>{v}</button>
                  ))}
                </div>
                <div className="flex flex-wrap gap-2 items-center">
                  <span className="text-sm text-gray-300 w-28">Color Blind</span>
                  {(['none','protanopia','deuteranopia','tritanopia','achroma'] as ColorBlind[]).map(c=>(
                    <button key={c} onClick={()=>setColorBlind(c)} className={`px-3 py-1.5 rounded text-sm border ${colorBlind===c?'bg-purple-600 text-white border-purple-500':'bg-gray-700/50 border-gray-600 hover:bg-gray-700'}`}>{c}</button>
                  ))}
                </div>
              </div>
              <div className="grid md:grid-cols-2 gap-3">
                <div className="flex flex-wrap gap-2 items-center">
                  <span className="text-sm text-gray-300 w-20">Hearing</span>
                  {(['normal','mute'] as HearingMode[]).map(h=>(
                    <button key={h} onClick={()=>setHearing(h)} className={`px-3 py-1.5 rounded text-sm border ${hearing===h?'bg-emerald-600 text-white border-emerald-500':'bg-gray-700/50 border-gray-600 hover:bg-gray-700'}`}>{h}</button>
                  ))}
                </div>
                <div className="flex flex-wrap gap-2 items-center">
                  <span className="text-sm text-gray-300 w-28">Motor</span>
                  {(['normal','singlehand'] as MotorMode[]).map(m=>(
                    <button key={m} onClick={()=>setMotor(m)} className={`px-3 py-1.5 rounded text-sm border ${motor===m?'bg-amber-600 text-white border-amber-500':'bg-gray-700/50 border-gray-600 hover:bg-gray-700'}`}>{m}</button>
                  ))}
                  <span className="text-xs text-gray-400 ml-2">Long-press to trigger in single-hand</span>
                </div>
              </div>
              <div className="flex flex-wrap gap-2 items-center">
                <span className="text-sm text-gray-300 w-20">Cognitive</span>
                {(['normal','dyslexia','simplify'] as CognitiveMode[]).map(c=>(
                  <button key={c} onClick={()=>setCognitive(c)} className={`px-3 py-1.5 rounded text-sm border ${cognitive===c?'bg-pink-600 text-white border-pink-500':'bg-gray-700/50 border-gray-600 hover:bg-gray-700'}`}>{c}</button>
                ))}
              </div>
            </div>
          )}

          <div className="max-w-7xl mx-auto cognitive-scope">
            <div className="grid lg:grid-cols-4 gap-6">
              <div className="lg:col-span-1 bg-gray-800 p-6 rounded-xl h-[calc(100vh-12rem)] sticky top-20 overflow-y-auto hidden md:block">
                <div className="flex items-center space-x-2 mb-4">
                  <Scan className="w-6 h-6 text-blue-400" />
                  <h2 className="text-xl font-semibold">Detectable Objects</h2>
                </div>
                <div className="space-y-2">
                  {indoorObjects.map((obj) => (
                    <div key={obj} className="flex items-center space-x-2 p-2 rounded bg-gray-700/50">
                      <div className="w-4 h-4 rounded-full" style={{ backgroundColor: getColorForClass(obj) }} />
                      <span className="capitalize">{obj}</span>
                    </div>
                  ))}
                </div>

                <div className="mt-6 bg-gray-800 p-4 rounded-xl space-y-3">
                  <h3 className="font-semibold">Hearing Simulation</h3>
                  <div className="bg-gray-700/50 p-3 rounded-lg flex items-center gap-3">
                    <span className="text-sm text-gray-300">Sample Audio</span>
                    <audio ref={audioRef} src="/audio/sample.mp3" controls className="w-full" />
                  </div>
                  <p className="text-xs text-gray-400">Live captions appear when hearing is mute.</p>
                </div>
              </div>

              <div className="lg:col-span-3 space-y-6">
                <div className="relative rounded-xl overflow-hidden bg-black h-[calc(100vh-12rem)]">
                  <Webcam
                    ref={webcamRef}
                    onUserMedia={handleUserMedia}
                    className={`w-full h-full object-contain ${visualClass}`}
                    mirrored
                    audio={false}
                  />
                  <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none" />

                  {visual === 'central' && (
                    <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                      <div className="w-64 h-64 rounded-full" style={{ boxShadow: '0 0 0 9999px rgba(0,0,0,0.9)' }} />
                    </div>
                  )}

                  {!isCameraReady && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-800">
                      <p className="text-lg">Please allow camera access...</p>
                    </div>
                  )}
                  {!model && isCameraReady && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-800/50">
                      <p className="text-lg">Loading object detection model...</p>
                    </div>
                  )}

                  <div className="absolute bottom-4 left-4 right-4 flex flex-col gap-1 items-stretch">
                    {(hearing !== 'normal' ? hearingCaptions : detCaptions).map((c, i) => (
                      <div key={i} className="bg-black/70 px-3 py-2 rounded text-sm border border-white/10">{c}</div>
                    ))}
                  </div>
                </div>

                {isDetectionActive && detectedObjects.length > 0 && (
                  <div className="bg-gray-800 p-6 rounded-xl">
                    <div className="flex items-center space-x-2 mb-4">
                      <Scan className="w-6 h-6 text-blue-400" />
                      <h2 className="text-xl font-semibold">Detection Results</h2>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {detectedObjects.map((obj, index) => (
                        <div
                          key={index}
                          className="bg-gray-700/50 p-4 rounded-lg"
                          style={{ borderLeft: `4px solid ${getColorForClass(obj.class)}` }}
                        >
                          <h3 className="font-medium capitalize">{obj.class}</h3>
                          <p className="text-sm text-gray-400">Confidence: {(obj.score * 100).toFixed(1)}%</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>
    </>
  );
}

export default App;
