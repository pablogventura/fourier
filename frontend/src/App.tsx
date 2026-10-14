import { useEffect, useMemo, useRef, useState } from "react"
import { analyzeContour, exportArtifact, importFlc1File, loadFile, loadMetrics, loadPreset, loadStroke, saveBlob } from "./engine/lab.ts"
import type { ExportRequest } from "./engine/export.ts"
import { ContourStage } from "./components/ContourStage.tsx"
import { EpicycleCanvas } from "./components/EpicycleCanvas.tsx"
import { JpegLab } from "./components/JpegLab.tsx"
import { Plot } from "./components/Plot.tsx"
import { plotLayout } from "./components/plotTheme.ts"
import { selectCoefficients, type Strategy } from "./math/series.ts"
import type { Appearance, ContourBundle, DrawMode, FourierResult, Metrics } from "./types.ts"

const PRESETS = [
  ["circle", "Círculo"],
  ["square", "Cuadrado"],
  ["star", "Estrella"],
  ["heart", "Corazón"],
  ["hummingbird", "Colibrí"],
] as const

const initialLook: Appearance = {
  background: "#0c0e12",
  original: "#a9b6c6",
  reconstruction: "#8ef0d0",
  circles: "#e4b15a",
  strokeWidth: 1.6,
  showCircles: true,
  showVectors: true,
  showGuides: true,
  showOriginal: true,
  colorByError: true,
}

type Raster = {
  threshold: number
  blurSigma: number
  minLength: number
  method: "binary" | "canny"
  polarity: "auto" | "bright" | "dark"
  sampleCount: number
}

export default function App() {
  const [tab, setTab] = useState<"contours" | "jpeg">("contours")
  const [bundle, setBundle] = useState<ContourBundle | null>(null)
  const [selectedId, setSelectedId] = useState(0)
  const [spectrum, setSpectrum] = useState<FourierResult | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [center, setCenter] = useState(true)
  const [strategy, setStrategy] = useState<Strategy>("symmetric")
  const [components, setComponents] = useState(31)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(0.18)
  const [resetToken, setResetToken] = useState(0)
  const [mode, setMode] = useState<"complex" | "cartesian">("complex")
  const [look, setLook] = useState<Appearance>(initialLook)
  const [raster, setRaster] = useState<Raster>({
    threshold: 128,
    blurSigma: 1,
    minLength: 24,
    method: "binary",
    polarity: "auto",
    sampleCount: 512,
  })
  const [drawing, setDrawing] = useState(false)
  const [drawMode, setDrawMode] = useState<DrawMode>("image")
  const [numberFormat, setNumberFormat] = useState<"float64" | "float32" | "int16">("float32")
  const [quantStep, setQuantStep] = useState(0.05)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const requestRef = useRef(0)
  const fileRef = useRef<File | null>(null)
  const presetRef = useRef("heart")
  const strokeRef = useRef<Array<[number, number]> | null>(null)
  const rasterRef = useRef(raster)
  const skipSampleReload = useRef(true)
  rasterRef.current = raster

  const contour = bundle?.contours.find((item) => item.id === selectedId) ?? bundle?.contours[0] ?? null

  async function refresh() {
    const params = rasterRef.current
    const request = requestRef.current + 1
    requestRef.current = request
    setBusy(true)
    setError(null)
    try {
      const next = fileRef.current
        ? await loadFile(fileRef.current, params)
        : strokeRef.current
          ? loadStroke(strokeRef.current, params.sampleCount, null)
          : loadPreset(presetRef.current, params.sampleCount)
      if (request !== requestRef.current) return
      setBundle(next)
      setSelectedId(next.contours[0]?.id ?? 0)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "No se pudo cargar el contorno.")
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    void refresh()
    // La carga inicial usa el corazón.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!fileRef.current) return
    const timer = window.setTimeout(() => void refresh(), 280)
    return () => window.clearTimeout(timer)
    // refresh lee el archivo y los parámetros desde refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raster.threshold, raster.blurSigma, raster.minLength, raster.method, raster.polarity])

  useEffect(() => {
    if (skipSampleReload.current) {
      skipSampleReload.current = false
      return
    }
    const timer = window.setTimeout(() => void refresh(), 280)
    return () => window.clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raster.sampleCount])

  useEffect(() => {
    if (!contour) return
    let cancelled = false
    void Promise.resolve()
      .then(() => analyzeContour(contour.points, contour.closed, center))
      .then((result) => {
        if (!cancelled) setSpectrum(result)
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "Falló la FFT.")
      })
    return () => {
      cancelled = true
    }
  }, [contour, center])

  useEffect(() => {
    if (!contour) return
    const timer = window.setTimeout(() => {
      void loadMetrics({
        points: contour.points,
        closed: contour.closed,
        center,
        strategy,
        component_count: components,
        number_format: numberFormat,
        quant_step: quantStep,
      })
        .then(setMetrics)
        .catch(() => setMetrics(null))
    }, 220)
    return () => window.clearTimeout(timer)
  }, [contour, center, strategy, components, numberFormat, quantStep])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.code !== "Space") return
      const tag = (event.target as HTMLElement | null)?.tagName
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return
      event.preventDefault()
      setPlaying((value) => !value)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  const selected = useMemo(
    () => (spectrum ? selectCoefficients(spectrum.coefficients, strategy, components) : []),
    [spectrum, strategy, components],
  )
  const selectedX = useMemo(
    () => (spectrum ? selectCoefficients(spectrum.x_coefficients, strategy, components) : []),
    [spectrum, strategy, components],
  )
  const selectedY = useMemo(
    () => (spectrum ? selectCoefficients(spectrum.y_coefficients, strategy, components) : []),
    [spectrum, strategy, components],
  )

  const spectrumPlot = useMemo(() => {
    if (!spectrum) return { data: [], layout: plotLayout }
    const kept = new Set(selected.map((item) => item.frequency))
    return {
      data: [
        {
          type: "bar",
          x: spectrum.coefficients.map((item) => item.frequency),
          y: spectrum.coefficients.map((item) => item.amplitude),
          marker: {
            color: spectrum.coefficients.map((item) => (kept.has(item.frequency) ? "#e4b15a" : "#3d4c60")),
          },
        },
      ],
      layout: { ...plotLayout, xaxis: { ...plotLayout.xaxis, title: { text: "frecuencia k" } }, yaxis: { ...plotLayout.yaxis, title: { text: "|c_k|" } } },
    }
  }, [spectrum, selected])

  const ratePlot = useMemo(() => {
    if (!metrics) return { data: [], layout: plotLayout }
    return {
      data: [
        {
          type: "scatter",
          mode: "lines+markers",
          x: metrics.curve.map((item) => item.encoded_bytes),
          y: metrics.curve.map((item) => item.rmse),
          text: metrics.curve.map((item) => `${item.component_count} componentes`),
          hoverinfo: "text+x+y",
          line: { color: "#8ef0d0" },
          marker: { color: "#e4b15a", size: 6 },
        },
      ],
      layout: {
        ...plotLayout,
        xaxis: { ...plotLayout.xaxis, title: { text: "bytes del archivo FLC1" } },
        yaxis: { ...plotLayout.yaxis, title: { text: "RMSE" } },
      },
    }
  }, [metrics])

  function compressionBody(extra: Record<string, unknown> = {}) {
    if (!contour) return null
    return {
      points: contour.points,
      closed: contour.closed,
      center,
      strategy,
      component_count: components,
      number_format: numberFormat,
      quant_step: quantStep,
      appearance: {
        background: look.background,
        original: look.original,
        reconstruction: look.reconstruction,
        circles: look.circles,
        stroke_width: look.strokeWidth,
        show_circles: look.showCircles,
        show_vectors: look.showVectors,
        show_guides: look.showGuides,
        show_original: look.showOriginal,
        color_by_error: look.colorByError,
        mode,
      },
      frames: 72,
      width: 640,
      height: 480,
      fps: 12,
      ...extra,
    }
  }

  async function save(kind: string) {
    const body = compressionBody()
    if (!body || !contour) return
    setBusy(true)
    setError(null)
    try {
      const artifact = await exportArtifact(kind, body as ExportRequest)
      saveBlob(artifact.blob, artifact.filename)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "No se pudo exportar.")
    } finally {
      setBusy(false)
    }
  }

  const warnings = [...(bundle?.warnings ?? []), ...(spectrum?.warnings ?? [])]
  const used = selected.length

  return (
    <div className="app">
      <header>
        <div className="brand">
          <strong>FourierLab</strong>
          <span>contornos, epiciclos y compresión</span>
        </div>
        <nav>
          <button type="button" className={tab === "contours" ? "active" : ""} onClick={() => setTab("contours")}>Contornos</button>
          <button type="button" className={tab === "jpeg" ? "active" : ""} onClick={() => setTab("jpeg")}>DCT / JPEG</button>
        </nav>
        <p className="formula">z = x - i y, &nbsp; c_k = FFT(z) / N, &nbsp; z(t) = Σ c_k exp(2 π i k t)</p>
      </header>
      {tab === "jpeg" ? (
        <JpegLab />
      ) : (
        <>
          <aside className="panel left">
            <h2>Fuente</h2>
            <div className="chips">
              {PRESETS.map(([id, title]) => (
                <button
                  key={id}
                  type="button"
                  className={presetRef.current === id && !fileRef.current && !strokeRef.current ? "active" : ""}
                  onClick={() => {
                    presetRef.current = id
                    fileRef.current = null
                    strokeRef.current = null
                    setDrawing(false)
                    void refresh()
                  }}
                >
                  {title}
                </button>
              ))}
            </div>
            <label className="file">
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp,image/svg+xml,.svg"
                onChange={(event) => {
                  const file = event.target.files?.[0]
                  if (!file) return
                  fileRef.current = file
                  strokeRef.current = null
                  presetRef.current = ""
                  setDrawMode("image")
                  void refresh()
                }}
              />
              Cargar PNG, JPG, WebP o SVG
            </label>
            <label>
              Muestras N
              <input type="range" min={64} max={2048} step={64} value={raster.sampleCount} onChange={(event) => setRaster({ ...raster, sampleCount: Number(event.target.value) })} />
              <span>{raster.sampleCount}</span>
            </label>
            <label>
              Umbral
              <input type="range" min={0} max={255} value={raster.threshold} onChange={(event) => setRaster({ ...raster, threshold: Number(event.target.value) })} />
              <span>{raster.threshold}</span>
            </label>
            <label>
              Desenfoque
              <input type="range" min={0} max={8} step={0.1} value={raster.blurSigma} onChange={(event) => setRaster({ ...raster, blurSigma: Number(event.target.value) })} />
              <span>{raster.blurSigma.toFixed(1)}</span>
            </label>
            <label>
              Longitud mínima
              <input type="range" min={0} max={400} value={raster.minLength} onChange={(event) => setRaster({ ...raster, minLength: Number(event.target.value) })} />
              <span>{raster.minLength}</span>
            </label>
            <label>
              Método
              <select value={raster.method} onChange={(event) => setRaster({ ...raster, method: event.target.value as Raster["method"] })}>
                <option value="binary">Umbral</option>
                <option value="canny">Canny</option>
              </select>
            </label>
            <label>
              Polaridad
              <select value={raster.polarity} onChange={(event) => setRaster({ ...raster, polarity: event.target.value as Raster["polarity"] })}>
                <option value="auto">Automática</option>
                <option value="dark">Figura oscura</option>
                <option value="bright">Figura clara</option>
              </select>
            </label>
            <div className="row">
              <button type="button" className={drawing ? "active" : ""} onClick={() => setDrawing((value) => !value)}>
                {drawing ? "Dibujando" : "Dibujar"}
              </button>
              <button type="button" className={drawMode === "image" ? "active" : ""} onClick={() => setDrawMode("image")}>Sobre la imagen</button>
              <button type="button" className={drawMode === "blank" ? "active" : ""} onClick={() => setDrawMode("blank")}>Lienzo vacío</button>
            </div>
            <ContourStage
              preview={bundle?.preview ?? null}
              contours={bundle?.contours ?? []}
              selectedId={contour?.id ?? -1}
              onSelect={setSelectedId}
              drawing={drawing}
              drawMode={drawMode}
              onStroke={(points) => {
                strokeRef.current = points
                fileRef.current = null
                presetRef.current = ""
                setDrawing(false)
                void refresh()
              }}
            />
            <ul className="contours">
              {(bundle?.contours ?? []).map((item) => (
                <li key={item.id}>
                  <button type="button" className={item.id === contour?.id ? "active" : ""} onClick={() => setSelectedId(item.id)}>
                    {item.label} {item.id + 1}
                    <small>{item.closed ? "cerrado" : "abierto"} · {item.perimeter.toFixed(0)} px</small>
                  </button>
                </li>
              ))}
            </ul>
            {warnings.slice(0, 4).map((warning) => <p key={warning} className="note">{warning}</p>)}
            {error && <p className="error">{error}</p>}
            {busy && <p className="note">Procesando...</p>}
          </aside>
          <main className="center">
            {contour && spectrum ? (
              <EpicycleCanvas
                points={contour.points}
                closed={contour.closed}
                offset={spectrum.offset}
                coefficients={selected}
                xCoefficients={selectedX}
                yCoefficients={selectedY}
                mode={mode}
                playing={playing}
                speed={speed}
                resetToken={resetToken}
                appearance={look}
              />
            ) : (
              <div className="empty">Cargando el contorno...</div>
            )}
            <p className="caption">
              y de la imagen apunta hacia abajo y el imaginario hacia arriba, así una frecuencia positiva gira en antihorario.
              Arrastrá para mover, la rueda hace zoom y un doble clic encuadra de nuevo.
            </p>
          </main>
          <aside className="panel right">
            <h2>Espectro</h2>
            <p className="note">En ámbar, los {used} coeficientes que entran en la suma. El resto se calculó y queda fuera.</p>
            <Plot data={spectrumPlot.data} layout={spectrumPlot.layout} />
            <h2>Métricas del contorno</h2>
            {metrics ? (
              <div className="metrics">
                <div className="metric"><span>RMSE</span><strong>{metrics.rmse.toFixed(4)}</strong></div>
                <div className="metric"><span>Error máximo</span><strong>{metrics.max_error.toFixed(4)}</strong></div>
                <div className="metric"><span>Hausdorff</span><strong>{metrics.hausdorff.toFixed(4)}</strong></div>
                <div className="metric"><span>Puntos crudos</span><strong>{metrics.raw_points_bytes} B</strong></div>
                <div className="metric"><span>Coeficientes crudos</span><strong>{metrics.raw_coefficients_bytes} B</strong></div>
                <div className="metric"><span>Archivo FLC1</span><strong>{metrics.encoded_bytes} B</strong></div>
                <div className="metric"><span>Crudo / archivo</span><strong>{metrics.ratio_vs_points?.toFixed(2) ?? "-"}</strong></div>
              </div>
            ) : (
              <p className="note">Las métricas aparecen cuando el contorno está listo.</p>
            )}
            <p className="note">
              Los bytes crudos son N muestras o N coeficientes en float64, sin cabecera. El FLC1 es el archivo de verdad.
              Esa curva no es la imagen: un contorno de Fourier no comprime el mapa de bits.
            </p>
            <h2>Tasa y distorsión</h2>
            <Plot data={ratePlot.data} layout={ratePlot.layout} />
          </aside>
          <footer className="deck">
            <div className="transport">
              <button type="button" className="primary" onClick={() => setPlaying((value) => !value)}>{playing ? "Pausa" : "Reproducir"}</button>
              <button type="button" onClick={() => { setPlaying(false); setResetToken((value) => value + 1) }}>Reiniciar</button>
              <label>
                Velocidad
                <input type="range" min={0.04} max={0.8} step={0.02} value={speed} onChange={(event) => setSpeed(Number(event.target.value))} />
              </label>
              <label>
                Componentes
                <input type="range" min={1} max={500} value={components} onChange={(event) => setComponents(Number(event.target.value))} />
                <span>{used}</span>
              </label>
              <label>
                Estrategia
                <select value={strategy} onChange={(event) => setStrategy(event.target.value as Strategy)}>
                  <option value="symmetric">Simétrica, de -K a K</option>
                  <option value="magnitude">Mayor magnitud</option>
                </select>
              </label>
              <label>
                Modo
                <select value={mode} onChange={(event) => setMode(event.target.value as "complex" | "cartesian")}>
                  <option value="complex">Epiciclos complejos</option>
                  <option value="cartesian">Cadenas x e y</option>
                </select>
              </label>
              <label className="check">
                <input type="checkbox" checked={center} onChange={(event) => setCenter(event.target.checked)} />
                Centrar
              </label>
            </div>
            <div className="transport">
              <label className="check"><input type="checkbox" checked={look.showCircles} onChange={(event) => setLook({ ...look, showCircles: event.target.checked })} />Círculos</label>
              <label className="check"><input type="checkbox" checked={look.showVectors} onChange={(event) => setLook({ ...look, showVectors: event.target.checked })} />Vectores</label>
              <label className="check"><input type="checkbox" checked={look.showGuides} onChange={(event) => setLook({ ...look, showGuides: event.target.checked })} />Guías</label>
              <label className="check"><input type="checkbox" checked={look.showOriginal} onChange={(event) => setLook({ ...look, showOriginal: event.target.checked })} />Original</label>
              <label className="check"><input type="checkbox" checked={look.colorByError} onChange={(event) => setLook({ ...look, colorByError: event.target.checked })} />Color de error</label>
              <label className="color">Fondo<input type="color" value={look.background} onChange={(event) => setLook({ ...look, background: event.target.value })} /></label>
              <label className="color">Curva<input type="color" value={look.reconstruction} onChange={(event) => setLook({ ...look, reconstruction: event.target.value })} /></label>
              <label className="color">Círculos<input type="color" value={look.circles} onChange={(event) => setLook({ ...look, circles: event.target.value })} /></label>
              <label>
                Grosor
                <input type="range" min={0.6} max={6} step={0.2} value={look.strokeWidth} onChange={(event) => setLook({ ...look, strokeWidth: Number(event.target.value) })} />
              </label>
            </div>
            <div className="transport">
              <label>
                Precisión
                <select value={numberFormat} onChange={(event) => setNumberFormat(event.target.value as "float64" | "float32" | "int16")}>
                  <option value="float64">float64</option>
                  <option value="float32">float32</option>
                  <option value="int16">int16 cuantizado</option>
                </select>
              </label>
              <label>
                Paso
                <input type="number" min={0.0001} step={0.01} value={quantStep} disabled={numberFormat !== "int16"} onChange={(event) => setQuantStep(Math.max(0.0001, Number(event.target.value) || 0.0001))} />
              </label>
              <button type="button" onClick={() => void save("svg")}>SVG</button>
              <button type="button" onClick={() => void save("png")}>PNG</button>
              <button type="button" onClick={() => void save("gif")}>GIF</button>
              <button type="button" onClick={() => void save("webm")}>WebM</button>
              <button type="button" onClick={() => void save("csv")}>CSV</button>
              <button type="button" onClick={() => void save("json")}>JSON</button>
              <button type="button" onClick={() => void save("flc1")}>FLC1</button>
              <button type="button" onClick={() => void save("metrics")}>Métricas</button>
              <label className="file slim">
                <input
                  type="file"
                  accept=".flc1"
                  onChange={(event) => {
                    const file = event.target.files?.[0]
                    if (!file) return
                    void importFlc1File(file).then((payload) => {
                      requestRef.current += 1
                      strokeRef.current = payload.points
                      fileRef.current = null
                      presetRef.current = ""
                      setCenter(payload.centered)
                      setBundle({
                        source: "import",
                        width: 0,
                        height: 0,
                        preview: null,
                        warnings: [payload.note],
                        contours: [{
                          id: 0,
                          closed: payload.closed,
                          is_hole: false,
                          perimeter: 0,
                          sample_count: payload.sample_count,
                          points: payload.points,
                          bbox: [0, 0, 0, 0],
                          label: "importado",
                        }],
                      })
                      setSelectedId(0)
                    }).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "No se pudo importar."))
                  }}
                />
                Importar FLC1
              </label>
            </div>
          </footer>
        </>
      )}
    </div>
  )
}
