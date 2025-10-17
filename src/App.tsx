import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { Camera, Scan, Settings, ChevronDown } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import AccessibilityPanel, {
  VisualLevel, ColorBlind, HearingMode, MotorMode, CognitiveMode
} from './components/AccessibilityPanel';
import ColorBlindSVGDefs from './components/ColorBlindSVGDefs';
import useAudioFilter from './hooks/useAudioFilter';

declare global {
  interface Window {
    webkitSpeechRecognition?: any;
    SpeechRecognition?: any;
  }
}

function App() {
  const [isDetectionActive, setIsDetectionActive] = useState(false);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [detectedObjects, setDetectedObjects] = useState<Array<{ class: string; score: number }>>([]);
  const previousObjectsRef = useRef<Set<string>>(new Set());
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

  const [captions, setCaptions] = useState<string[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  useAudioFilter(audioRef.current, hearing === 'lowpass' ? 'lowpass' : hearing === 'highpass' ? 'highpass' : 'none');

  const [micStream, setMicStream] = useState<MediaStream | null>(null);
  const micCtxRef = useRef<AudioContext | null>(null);
  const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const micFilterRef = useRef<BiquadFilterNode | null>(null);
  const micGainRef = useRef<GainNode | null>(null);

  const [controlsOpen, setControlsOpen] = useState(false);

  const ensureMic = useCallback(async () => {
    if (micStream) return micStream;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    setMicStream(stream);
    return stream;
  }, [micStream]);

  const wireMicGraph = useCallback((mode: HearingMode) => {
    if (!micStream) return;
    if (!micCtxRef.current) micCtxRef.current = new AudioContext();
    const ctx = micCtxRef.current;

    if (micSourceRef.current) micSourceRef.current.disconnect();
    if (micFilterRef.current) micFilterRef.current.disconnect();
    if (micGainRef.current) micGainRef.current.disconnect();

    const source = ctx.createMediaStreamSource(micStream);
    const filter = ctx.createBiquadFilter();
    const gain = ctx.createGain();

    if (mode === 'lowpass') {
      filter.type = 'lowpass';
      filter.frequency.value = 1200;
      gain.gain.value = 1.0;
      source.connect(filter).connect(gain).connect(ctx.destination);
    } else if (mode === 'highpass') {
      filter.type = 'highpass';
      filter.frequency.value = 1200;
      gain.gain.value = 1.0;
      source.connect(filter).connect(gain).connect(ctx.destination);
    } else if (mode === 'mute') {
      filter.type = 'allpass';
      gain.gain.value = 0.0;
      source.connect(filter).connect(gain).connect(ctx.destination);
    } else {
      filter.type = 'allpass';
      gain.gain.value = 0.0;
      source.connect(filter).connect(gain).disconnect();
    }

    micSourceRef.current = source;
    micFilterRef.current = filter;
    micGainRef.current = gain;
  }, [micStream]);

  const recognitionRef = useRef<any>(null);
  const [asrActive, setAsrActive] = useState(false);

  useEffect(() => {
    if (!audioRef.current) return;
    if (hearing === 'normal') {
      audioRef.current.muted = false;
    } else {
      audioRef.current.muted = true;
    }
  }, [hearing]);

  useEffect(() => {
    (async () => {
      if (hearing !== 'normal') {
        try {
          await ensureMic();
          await micCtxRef.current?.resume();
          wireMicGraph(hearing);
        } catch (e) {
          console.warn('Microphone access failed or blocked.', e);
        }
      } else {
        if (micSourceRef.current) micSourceRef.current.disconnect();
        if (micFilterRef.current) micFilterRef.current.disconnect();
        if (micGainRef.current) micGainRef.current.disconnect();
      }
    })();
  }, [hearing, ensureMic, wireMicGraph]);

  const startASR = useCallback(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;
    if (recognitionRef.current) return;
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = 'en-US';
    rec.onresult = (e: any) => {
      let s = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        s += e.results[i][0].transcript;
      }
      if (s.trim().length > 0) {
        setCaptions((prev) => [s.trim(), ...prev].slice(0, 8));
      }
    };
    rec.onend = () => {
      recognitionRef.current = null;
      if (asrActive) startASR();
    };
    recognitionRef.current = rec;
    rec.start();
  }, [asrActive]);

  useEffect(() => {
    if (hearing === 'normal') {
      setAsrActive(false);
      recognitionRef.current?.stop();
      recognitionRef.current = null;
      return;
    }
    setAsrActive(true);
    startASR();
    return () => {
      recognitionRef.current?.stop();
      recognitionRef.current = null;
    };
  }, [hearing, startASR]);

  const speak = useCallback((text: string) => {
    if (isDetectionActive && isDetectionRunningRef.current) {
      setCaptions((prev) => [text, ...prev].slice(0, 8));
      if (hearing !== 'mute') {
        const utterance = new SpeechSynthesisUtterance(text);
        speechSynthesis.speak(utterance);
      }
    }
  }, [hearing, isDetectionActive]);

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
        console.error('TF init or model load error:', error);
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
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
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

      const currentObjects = new Set(predictions.map(p => p.class));
      currentObjects.forEach(obj => {
        if (!previousObjectsRef.current.has(obj)) speak(`Detected ${obj}`);
      });
      previousObjectsRef.current = currentObjects;

      setDetectedObjects(predictions.map(p => ({ class: p.class, score: p.score })));

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
  const colorClass = colorBlind === 'none' ? '' : `cb-${colorBlind}`;

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
      <div className={`min-h-screen bg-gray-900 text-white ${colorClass} ${motor === 'singlehand' ? 'motor-singlehand' : ''} ${cognitive === 'simplify' ? 'cog-simplify' : ''}`}>
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
                  onClick={() => setControlsOpen((v) => !v)}
                  className="px-3 py-1.5 rounded-md text-sm font-medium bg-gray-700 hover:bg-gray-600 flex items-center gap-2"
                >
                  <Settings className="w-4 h-4" />
                  <span>Modes</span>
                  <ChevronDown className={`w-4 h-4 transition ${controlsOpen ? 'rotate-180' : ''}`} />
                </button>
              </div>
            </div>
            {controlsOpen && (
              <div className="bg-gray-800/95">
                <div className="max-w-7xl mx-auto px-4 py-3">
                  <AccessibilityPanel
                    visual={visual} setVisual={setVisual}
                    color={colorBlind} setColor={setColorBlind}
                    hearing={hearing} setHearing={setHearing}
                    motor={motor} setMotor={setMotor}
                    cognitive={cognitive} setCognitive={setCognitive}
                  />
                </div>
              </div>
            )}
          </div>
        </header>

        <main className="container mx-auto px-4 pt-16 pb-6">
          <div className="max-w-7xl mx-auto cognitive-scope">
            <div className="grid lg:grid-cols-4 gap-6">
              <div className="lg:col-span-1 bg-gray-800 p-6 rounded-xl h-[calc(100vh-12rem)] sticky top-20 overflow-y-auto">
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
                  <p className="text-xs text-gray-400">Microphone processing starts when mode ≠ normal. Live captions appear while any hearing mode is active.</p>
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
                    {captions.map((c, i) => (
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
