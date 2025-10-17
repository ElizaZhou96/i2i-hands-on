// src/components/ColorBlindSVGDefs.tsx
export default function ColorBlindSVGDefs() {
    return (
      <svg className="hidden">
        {/* Deuteranopia (green cone absent) */}
        <filter id="cb-deuteranopia">
          <feColorMatrix type="matrix" values="
            0.625,0.375,0,0,0
            0.7,  0.3,  0,0,0
            0,    0.3,  0.7,0,0
            0,    0,    0,1,0" />
        </filter>
  
        {/* Protanopia (red cone absent) */}
        <filter id="cb-protanopia">
          <feColorMatrix type="matrix" values="
            0.567,0.433,0,0,0
            0.558,0.442,0,0,0
            0,    0.242,0.758,0,0
            0,    0,    0,1,0" />
        </filter>
  
        {/* Tritanopia (blue-yellow) */}
        <filter id="cb-tritanopia">
          <feColorMatrix type="matrix" values="
            0.95, 0.05,  0,0,0
            0,    0.433,0.567,0,0
            0,    0.475,0.525,0,0
            0,    0,    0,1,0" />
        </filter>
  
        {/* Achromatopsia (monochrome) */}
        <filter id="cb-achroma">
          <feColorMatrix type="matrix" values="
            0.299,0.587,0.114,0,0
            0.299,0.587,0.114,0,0
            0.299,0.587,0.114,0,0
            0,    0,    0,1,0" />
        </filter>
      </svg>
    );
  }
  