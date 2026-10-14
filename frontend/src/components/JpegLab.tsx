import { useEffect, useRef, useState } from "react"
import { loadBasis, loadJpeg, reconstructJpeg } from "../engine/lab.ts"
import type { JpegResult } from "../types.ts"

type Edit = { u: number; v: number; value: number }

export function JpegLab() {
  const [imageId, setImageId] = useState<string | null>(null)
  const [patterns, setPatterns] = useState<number[][][]>([])
  const [quality, setQuality] = useState(50)
  const [cutoff, setCutoff] = useState(14)
  const [grayscale, setGrayscale] = useState(false)
  const [channel, setChannel] = useState(0)
  const [block, setBlock] = useState({ row: 0, col: 0 })
  const [edits, setEdits] = useState<Edit[]>([])
  const [selected, setSelected] = useState({ u: 0, v: 0 })
  const [draft, setDraft] = useState<string | null>(null)
  const [result, setResult] = useState<JpegResult | null>(null)
  const [note, setNote] = useState("Cargá un PNG, JPG o WebP. El SVG no entra en esta vista.")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    void loadBasis().then((payload) => setPatterns(payload.patterns)).catch(() => setPatterns([]))
  }, [])

  useEffect(() => {
    if (!imageId) return
    const timer = window.setTimeout(() => {
      void reconstructJpeg({
        image_id: imageId,
        quality,
        cutoff,
        grayscale,
        block_row: block.row,
        block_col: block.col,
        channel,
        edits,
      })
        .then((payload) => {
          setResult(payload)
          setError(null)
        })
        .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "No se pudo reconstruir."))
    }, 180)
    return () => window.clearTimeout(timer)
  }, [imageId, quality, cutoff, grayscale, channel, block, edits])

  const cell = result?.block?.dequantized?.[selected.v]?.[selected.u]
  const shown = draft ?? (typeof cell === "number" ? cell.toFixed(2) : "0")

  function pickCell(u: number, v: number) {
    setSelected({ u, v })
    setDraft(null)
  }

  function commitEdit() {
    if (draft === null) return
    const value = Number(draft)
    if (!Number.isFinite(value)) return
    setDraft(null)
    setEdits((current) => {
      const rest = current.filter((item) => item.u !== selected.u || item.v !== selected.v)
      return [...rest, { u: selected.u, v: selected.v, value }]
    })
  }

  return (
    <section className="jpeg">
      <div className="panel">
        <h2>Imagen</h2>
        <label className="file">
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (!file) return
              void loadJpeg(file)
                .then((payload) => {
                  setImageId(payload.image_id)
                  setNote(payload.note)
                  setEdits([])
                  setBlock({ row: 0, col: 0 })
                })
                .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "No se pudo cargar."))
            }}
          />
          Elegir imagen
        </label>
        <label>
          Calidad IJG
          <input type="range" min={1} max={100} value={quality} onChange={(event) => { setQuality(Number(event.target.value)); setEdits([]) }} />
          <span>{quality}</span>
        </label>
        <label>
          Corte u + v
          <input type="range" min={0} max={14} value={cutoff} onChange={(event) => { setCutoff(Number(event.target.value)); setEdits([]) }} />
          <span>{cutoff === 14 ? "todas" : `hasta ${cutoff}`}</span>
        </label>
        <label className="check">
          <input type="checkbox" checked={grayscale} onChange={(event) => { setGrayscale(event.target.checked); setEdits([]) }} />
          Escala de grises
        </label>
        {!grayscale && (
          <label>
            Canal
            <select value={channel} onChange={(event) => { setChannel(Number(event.target.value)); setEdits([]) }}>
              <option value={0}>R</option>
              <option value={1}>G</option>
              <option value={2}>B</option>
            </select>
          </label>
        )}
        <p className="note">{note}</p>
        <p className="note">
          Antes de la DCT se resta 128. La tabla es la de luminancia del JPEG, escalada como en el IJG.
          No hay YCbCr, ni 4:2:0, ni codificación entrópica.
        </p>
        {error && <p className="error">{error}</p>}
      </div>
      <div className="panel jpeg-main">
        <h2>Patrones base y bloque {block.col}, {block.row}</h2>
        <BasisGrid patterns={patterns} active={selected} onPick={pickCell} />
        {result?.block && (
          <div className="coeff-editor">
            <CoeffGrid
              values={result.block.dequantized}
              active={selected}
              onPick={pickCell}
            />
            <label>
              Coeficiente u={selected.u}, v={selected.v}
              <input value={shown} onChange={(event) => setDraft(event.target.value)} onBlur={commitEdit} onKeyDown={(event) => { if (event.key === "Enter") commitEdit() }} />
            </label>
            <button type="button" onClick={() => setEdits([])}>Restaurar bloque</button>
            <p className="note">El número que editás entra directo a la DCT inversa, después de cuantizar y de borrar las frecuencias altas.</p>
          </div>
        )}
      </div>
      <div className="panel">
        <h2>Comparación</h2>
        <div className="metric"><span>PSNR</span><strong>{result?.psnr == null ? "sin error medible" : `${result.psnr.toFixed(2)} dB`}</strong></div>
        <div className="metric"><span>SSIM</span><strong>{result ? result.ssim.toFixed(4) : "-"}</strong></div>
        <div className="jpeg-images">
          {result && (
            <>
              <figure>
                <img src={result.original} alt="Imagen original del laboratorio DCT" />
                <figcaption>Original</figcaption>
              </figure>
              <figure>
                <img
                  src={result.reconstructed}
                  alt="Imagen reconstruida"
                  onClick={(event) => {
                    const img = event.currentTarget
                    const rect = img.getBoundingClientRect()
                    const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight)
                    const ox = (rect.width - img.naturalWidth * scale) / 2
                    const oy = (rect.height - img.naturalHeight * scale) / 2
                    const x = (event.clientX - rect.left - ox) / scale
                    const y = (event.clientY - rect.top - oy) / scale
                    if (x < 0 || y < 0 || x >= img.naturalWidth || y >= img.naturalHeight) return
                    setBlock({ col: Math.floor(x / 8), row: Math.floor(y / 8) })
                    setEdits([])
                  }}
                />
                <figcaption>Reconstruida. Clic para elegir el bloque.</figcaption>
              </figure>
              <figure>
                <img src={result.difference} alt="Diferencia amplificada" />
                <figcaption>Diferencia x4</figcaption>
              </figure>
            </>
          )}
        </div>
      </div>
    </section>
  )
}

function BasisGrid({
  patterns,
  active,
  onPick,
}: {
  patterns: number[][][]
  active: { u: number; v: number }
  onPick: (u: number, v: number) => void
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || patterns.length !== 64) return
    const cell = 28
    const dpr = window.devicePixelRatio || 1
    canvas.width = 8 * cell * dpr
    canvas.height = 8 * cell * dpr
    canvas.style.width = `${8 * cell}px`
    canvas.style.height = `${8 * cell}px`
    const context = canvas.getContext("2d")
    if (!context) return
    context.setTransform(dpr, 0, 0, dpr, 0, 0)
    patterns.forEach((pattern, index) => {
      const u = index % 8
      const v = Math.floor(index / 8)
      let low = Infinity
      let high = -Infinity
      for (const row of pattern) for (const value of row) {
        low = Math.min(low, value)
        high = Math.max(high, value)
      }
      const span = high - low || 1
      for (let y = 0; y < 8; y += 1) {
        for (let x = 0; x < 8; x += 1) {
          const shade = Math.round(((pattern[y][x] - low) / span) * 255)
          context.fillStyle = `rgb(${shade}, ${shade}, ${shade})`
          context.fillRect(u * cell + x * (cell / 8), v * cell + y * (cell / 8), cell / 8 + 0.5, cell / 8 + 0.5)
        }
      }
      if (u === active.u && v === active.v) {
        context.strokeStyle = "#e4b15a"
        context.lineWidth = 2
        context.strokeRect(u * cell + 1, v * cell + 1, cell - 2, cell - 2)
      }
    })
  }, [patterns, active])

  return (
    <canvas
      ref={canvasRef}
      className="basis"
      onClick={(event) => {
        const rect = event.currentTarget.getBoundingClientRect()
        const u = Math.min(7, Math.floor(((event.clientX - rect.left) / rect.width) * 8))
        const v = Math.min(7, Math.floor(((event.clientY - rect.top) / rect.height) * 8))
        onPick(u, v)
      }}
    />
  )
}

function CoeffGrid({
  values,
  active,
  onPick,
}: {
  values: number[][]
  active: { u: number; v: number }
  onPick: (u: number, v: number) => void
}) {
  return (
    <div className="coeff-grid">
      {values.map((row, v) =>
        row.map((value, u) => (
          <button
            key={`${u}-${v}`}
            type="button"
            className={u === active.u && v === active.v ? "active" : ""}
            onClick={() => onPick(u, v)}
          >
            {value.toFixed(0)}
          </button>
        )),
      )}
    </div>
  )
}
