import { GIFEncoder, applyPalette, quantize } from "gifenc"
import type { Coefficient, Strategy } from "../math/series.ts"
import { axisChain, realValue, reconstructSource, selectCoefficients, sourceChain } from "../math/series.ts"
import { analyzeSourcePoints } from "./dft.ts"
import type { NumberFormat } from "./flc1.ts"

export type ExportAppearance = {
  background: string
  original: string
  reconstruction: string
  circles: string
  stroke_width: number
  show_circles: boolean
  show_vectors: boolean
  show_guides: boolean
  show_original: boolean
  color_by_error: boolean
  mode: "complex" | "cartesian"
}

export type ExportRequest = {
  points: Array<[number, number]>
  closed: boolean
  center: boolean
  strategy: Strategy
  component_count: number
  number_format: NumberFormat
  quant_step: number
  appearance: ExportAppearance
  frames: number
  width: number
  height: number
  fps: number
}

type Rgb = [number, number, number]

function parseHex(value: string): Rgb {
  const text = value.trim().replace("#", "")
  if (!/^[0-9a-fA-F]{6}$/.test(text)) throw new Error(`Color inválido: ${value}`)
  return [Number.parseInt(text.slice(0, 2), 16), Number.parseInt(text.slice(2, 4), 16), Number.parseInt(text.slice(4, 6), 16)]
}

function mix(t: number): Rgb {
  const clamped = Math.max(0, Math.min(1, t))
  const low: Rgb = [42, 111, 219]
  const high: Rgb = [232, 93, 76]
  return [
    Math.round(low[0] + (high[0] - low[0]) * clamped),
    Math.round(low[1] + (high[1] - low[1]) * clamped),
    Math.round(low[2] + (high[2] - low[2]) * clamped),
  ]
}

function css([r, g, b]: Rgb): string {
  return `rgb(${r} ${g} ${b})`
}

type Prepared = {
  selected: Coefficient[]
  xSelected: Coefficient[]
  ySelected: Coefficient[]
  offsetX: number
  offsetY: number
  reconstructed: Array<{ x: number; y: number }>
  errors: number[]
  peak: number
}

function prepare(request: ExportRequest): Prepared {
  const spectrum = analyzeSourcePoints(request.points, request.center)
  const selected = selectCoefficients(spectrum.coefficients, request.strategy, request.component_count)
  const reconstructed = reconstructSource(selected, spectrum.sampleCount, spectrum.offsetX, spectrum.offsetY)
  const errors = request.points.map(([x, y], index) => Math.hypot(x - reconstructed[index].x, y - reconstructed[index].y))
  return {
    selected,
    xSelected: selectCoefficients(spectrum.xCoefficients, request.strategy, request.component_count),
    ySelected: selectCoefficients(spectrum.yCoefficients, request.strategy, request.component_count),
    offsetX: spectrum.offsetX,
    offsetY: spectrum.offsetY,
    reconstructed,
    errors,
    peak: Math.max(1e-9, ...errors),
  }
}

function projectOf(request: ExportRequest, prepared: Prepared) {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  const consider = (x: number, y: number) => {
    minX = Math.min(minX, x)
    minY = Math.min(minY, y)
    maxX = Math.max(maxX, x)
    maxY = Math.max(maxY, y)
  }
  for (const [x, y] of request.points) consider(x, y)
  for (const point of prepared.reconstructed) consider(point.x, point.y)
  const spanX = Math.max(maxX - minX, 1)
  const spanY = Math.max(maxY - minY, 1)
  minX -= spanX * 0.35
  minY -= spanY * 0.35
  maxX += spanX * 0.35
  maxY += spanY * 0.35
  const scale = Math.min((request.width - 56) / (maxX - minX), (request.height - 56) / (maxY - minY))
  const originX = (request.width - (maxX - minX) * scale) / 2 - minX * scale
  const originY = (request.height - (maxY - minY) * scale) / 2 - minY * scale
  const rawMinY = Math.min(...request.points.map((point) => point[1]))
  const rawMinX = Math.min(...request.points.map((point) => point[0]))
  return {
    scale,
    originX,
    originY,
    yAnchor: rawMinY - spanY * 0.42,
    xAnchor: rawMinX - spanX * 0.42,
    map(x: number, y: number) {
      return { x: originX + x * scale, y: originY + y * scale }
    },
  }
}

function stroke(
  context: CanvasRenderingContext2D,
  points: Array<{ x: number; y: number }>,
  color: string,
  width: number,
  closed: boolean,
  map: (x: number, y: number) => { x: number; y: number },
) {
  if (points.length < 2) return
  context.beginPath()
  const first = map(points[0].x, points[0].y)
  context.moveTo(first.x, first.y)
  for (let index = 1; index < points.length; index += 1) {
    const point = map(points[index].x, points[index].y)
    context.lineTo(point.x, point.y)
  }
  if (closed) context.lineTo(first.x, first.y)
  context.strokeStyle = color
  context.lineWidth = Math.max(1, width)
  context.lineJoin = "round"
  context.lineCap = "round"
  context.stroke()
}

function drawCircles(
  context: CanvasRenderingContext2D,
  circles: Array<{ cx: number; cy: number; x: number; y: number; radius: number }>,
  map: (x: number, y: number) => { x: number; y: number },
  scale: number,
  color: string,
  showCircles: boolean,
  showVectors: boolean,
) {
  for (const circle of circles) {
    const center = map(circle.cx, circle.cy)
    const tip = map(circle.x, circle.y)
    const radius = circle.radius * Math.abs(scale)
    if (showCircles && radius >= 0.6) {
      context.beginPath()
      context.arc(center.x, center.y, radius, 0, Math.PI * 2)
      context.strokeStyle = color
      context.lineWidth = 1
      context.stroke()
    }
    if (showVectors) {
      context.beginPath()
      context.moveTo(center.x, center.y)
      context.lineTo(tip.x, tip.y)
      context.strokeStyle = color
      context.lineWidth = 2
      context.stroke()
    }
  }
}

function renderFrame(
  context: CanvasRenderingContext2D,
  request: ExportRequest,
  prepared: Prepared,
  t: number,
  trailCount: number,
) {
  const look = request.appearance
  const view = projectOf(request, prepared)
  context.fillStyle = look.background
  context.fillRect(0, 0, request.width, request.height)
  const source = request.points.map(([x, y]) => ({ x, y }))
  if (look.show_original) stroke(context, source, look.original, look.stroke_width, request.closed, view.map)
  if (!look.color_by_error) {
    const [r, g, b] = parseHex(look.reconstruction)
    stroke(context, prepared.reconstructed, css([Math.round(r * 0.45), Math.round(g * 0.45), Math.round(b * 0.45)]), look.stroke_width, request.closed, view.map)
  } else {
    context.lineWidth = Math.max(1, look.stroke_width)
    context.lineCap = "round"
    const steps = prepared.reconstructed.length - (request.closed ? 0 : 1)
    for (let index = 0; index < steps; index += 1) {
      const a = view.map(prepared.reconstructed[index].x, prepared.reconstructed[index].y)
      const next = prepared.reconstructed[(index + 1) % prepared.reconstructed.length]
      const b = view.map(next.x, next.y)
      context.beginPath()
      context.moveTo(a.x, a.y)
      context.lineTo(b.x, b.y)
      context.strokeStyle = css(mix(prepared.errors[index] / prepared.peak))
      context.stroke()
    }
  }
  const frames = Math.max(1, request.frames)
  const pen = penAt(request, prepared, t)
  const trail: Array<{ x: number; y: number }> = []
  for (let index = 0; index < trailCount; index += 1) trail.push(penAt(request, prepared, index / frames))
  stroke(context, trail, look.reconstruction, look.stroke_width + 1, false, view.map)
  const ink = look.circles
  if (look.mode === "complex") {
    drawCircles(context, sourceChain(prepared.selected, t, prepared.offsetX, prepared.offsetY), view.map, view.scale, ink, look.show_circles, look.show_vectors)
  } else {
    const xChain = axisChain(prepared.xSelected, t, prepared.offsetX, view.yAnchor, "x")
    const yChain = axisChain(prepared.ySelected, t, view.xAnchor, prepared.offsetY, "y")
    drawCircles(context, xChain, view.map, view.scale, ink, look.show_circles, look.show_vectors)
    drawCircles(context, yChain, view.map, view.scale, ink, look.show_circles, look.show_vectors)
    if (look.show_guides) {
      const target = view.map(pen.x, pen.y)
      context.strokeStyle = ink
      context.lineWidth = 1
      if (xChain.length) {
        const from = view.map(pen.x, xChain[xChain.length - 1].y)
        context.beginPath()
        context.moveTo(from.x, from.y)
        context.lineTo(target.x, target.y)
        context.stroke()
      }
      if (yChain.length) {
        const from = view.map(yChain[yChain.length - 1].x, pen.y)
        context.beginPath()
        context.moveTo(from.x, from.y)
        context.lineTo(target.x, target.y)
        context.stroke()
      }
    }
  }
  const dot = view.map(pen.x, pen.y)
  context.beginPath()
  context.fillStyle = look.reconstruction
  context.arc(dot.x, dot.y, 4, 0, Math.PI * 2)
  context.fill()
}

function penAt(request: ExportRequest, prepared: Prepared, t: number): { x: number; y: number } {
  if (request.appearance.mode === "cartesian") {
    return {
      x: prepared.offsetX + realValue(prepared.xSelected, t),
      y: prepared.offsetY + realValue(prepared.ySelected, t),
    }
  }
  const chain = sourceChain(prepared.selected, t, prepared.offsetX, prepared.offsetY)
  const tip = chain.at(-1)
  return tip ? { x: tip.x, y: tip.y } : { x: prepared.offsetX, y: prepared.offsetY }
}

function canvasOf(width: number, height: number): { canvas: HTMLCanvasElement; context: CanvasRenderingContext2D } {
  const canvas = document.createElement("canvas")
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext("2d", { willReadFrequently: true })
  if (!context) throw new Error("El navegador no pudo dibujar la exportación.")
  return { canvas, context }
}

export function renderSvg(
  original: Array<[number, number]>,
  reconstructed: Array<[number, number]>,
  closed: boolean,
  originalColor: string,
  reconstructionColor: string,
): string {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const [x, y] of [...original, ...reconstructed]) {
    minX = Math.min(minX, x)
    minY = Math.min(minY, y)
    maxX = Math.max(maxX, x)
    maxY = Math.max(maxY, y)
  }
  const pad = 12
  const tag = closed ? "polygon" : "polyline"
  const shape = (points: Array<[number, number]>, color: string) => {
    const coordinates = points.map(([x, y]) => `${x},${y}`).join(" ")
    return `  <${tag} points="${coordinates}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>`
  }
  return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="${minX - pad} ${minY - pad} ${Math.max(1, maxX - minX) + pad * 2} ${Math.max(1, maxY - minY) + pad * 2}">\n${shape(original, originalColor)}\n${shape(reconstructed, reconstructionColor)}\n</svg>\n`
}

export function renderPng(request: ExportRequest): Promise<Blob> {
  const prepared = prepare(request)
  const { canvas, context } = canvasOf(request.width, request.height)
  renderFrame(context, request, prepared, 0, 1)
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob)
      else reject(new Error("No se pudo armar el PNG."))
    }, "image/png")
  })
}

export function renderGif(request: ExportRequest): Blob {
  const prepared = prepare(request)
  const { context } = canvasOf(request.width, request.height)
  const frames = Math.max(1, Math.min(120, request.frames))
  renderFrame(context, { ...request, frames }, prepared, 0, 1)
  const first = context.getImageData(0, 0, request.width, request.height).data
  const palette = quantize(first, 128)
  const gif = GIFEncoder()
  const delay = Math.round(1000 / Math.max(1, request.fps))
  gif.writeFrame(applyPalette(first, palette), request.width, request.height, { palette, delay, repeat: 0 })
  for (let index = 1; index < frames; index += 1) {
    renderFrame(context, { ...request, frames }, prepared, index / frames, index + 1)
    const pixels = context.getImageData(0, 0, request.width, request.height).data
    gif.writeFrame(applyPalette(pixels, palette), request.width, request.height, { palette, delay })
  }
  gif.finish()
  const bytes = gif.bytes()
  const copy = new ArrayBuffer(bytes.byteLength)
  new Uint8Array(copy).set(bytes)
  return new Blob([copy], { type: "image/gif" })
}

export function renderWebm(request: ExportRequest): Promise<Blob> {
  if (typeof MediaRecorder === "undefined") throw new Error("Este navegador no graba video. Usá el GIF.")
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
    ? "video/webm;codecs=vp9"
    : MediaRecorder.isTypeSupported("video/webm")
      ? "video/webm"
      : ""
  if (!mime) throw new Error("Este navegador no graba WebM. Usá el GIF.")
  const prepared = prepare(request)
  const frames = Math.max(1, Math.min(120, request.frames))
  const { canvas, context } = canvasOf(request.width, request.height)
  const stream = canvas.captureStream(Math.max(1, request.fps))
  const recorder = new MediaRecorder(stream, { mimeType: mime })
  const chunks: Blob[] = []
  recorder.ondataavailable = (event) => {
    if (event.data.size) chunks.push(event.data)
  }
  return new Promise((resolve, reject) => {
    recorder.onerror = () => reject(new Error("No se pudo grabar el video. Usá el GIF."))
    recorder.onstop = () => resolve(new Blob(chunks, { type: "video/webm" }))
    recorder.start()
    let index = 0
    const timer = window.setInterval(() => {
      renderFrame(context, { ...request, frames }, prepared, index / frames, index + 1)
      const track = stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack | undefined
      track?.requestFrame()
      index += 1
      if (index >= frames) {
        window.clearInterval(timer)
        recorder.stop()
      }
    }, 1000 / Math.max(1, request.fps))
  })
}
