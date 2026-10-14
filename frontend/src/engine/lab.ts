import type { Strategy } from "../math/series.ts"
import { selectCoefficients } from "../math/series.ts"
import type { ContourBundle, FourierResult, JpegResult, Metrics } from "../types.ts"
import { analyzeSourcePoints } from "./dft.ts"
import { renderGif, renderPng, renderSvg, renderWebm, type ExportRequest } from "./export.ts"
import { decodeFlc1, encodeFlc1, reconstructDocument, type NumberFormat } from "./flc1.ts"
import { basisPatterns, reconstructStored, storeJpeg, type BlockEdit } from "./jpeg.ts"
import { measureCompression, reconstructedCurve } from "./metrics.ts"
import { extractRaster, type RasterParams } from "./raster.ts"
import { resampleContour, type Point } from "./resample.ts"
import { presetPoints } from "./shapes.ts"
import { extractSvgContours, makeContour } from "./svg.ts"

function span(points: Point[]): { width: number; height: number } {
  if (!points.length) return { width: 0, height: 0 }
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const [x, y] of points) {
    minX = Math.min(minX, x)
    minY = Math.min(minY, y)
    maxX = Math.max(maxX, x)
    maxY = Math.max(maxY, y)
  }
  return { width: maxX - minX, height: maxY - minY }
}

export function loadPreset(name: string, sampleCount: number): ContourBundle {
  const built = presetPoints(name, sampleCount)
  const size = span(built.points)
  const contour = makeContour(0, built.points, built.closed, false, name)
  contour.note = built.note
  return {
    source: "preset",
    width: size.width,
    height: size.height,
    preview: null,
    warnings: [
      "Una serie describe un solo contorno. Si la figura tiene huecos o partes separadas, cada una es otra señal.",
      built.note,
    ],
    contours: [contour],
  }
}

export function loadStroke(points: Point[], sampleCount: number, closed: boolean | null): ContourBundle {
  if (points.length > 20000) throw new Error("El trazo tiene demasiados puntos.")
  let isClosed = closed
  if (isClosed === null) {
    let minX = Infinity
    let minY = Infinity
    let maxX = -Infinity
    let maxY = -Infinity
    for (const [x, y] of points) {
      minX = Math.min(minX, x)
      minY = Math.min(minY, y)
      maxX = Math.max(maxX, x)
      maxY = Math.max(maxY, y)
    }
    const tolerance = Math.max(12, 0.04 * Math.max(maxX - minX, maxY - minY, 1))
    const start = points[0]
    const end = points[points.length - 1]
    isClosed = Math.hypot(start[0] - end[0], start[1] - end[1]) <= tolerance
  }
  const sampled = resampleContour(points, sampleCount, isClosed)
  const warnings = ["El trazo se remuestreó con arcos iguales. El parámetro no es el tiempo del dibujo."]
  if (!isClosed) {
    warnings.unshift("El trazo está abierto. La reconstrucción es periódica y cierra el salto: ahí aparece Gibbs.")
  }
  const size = span(sampled)
  return {
    source: "stroke",
    width: size.width,
    height: size.height,
    preview: null,
    warnings,
    contours: [makeContour(0, sampled, isClosed, false, "trazo")],
  }
}

function looksLikeSvg(name: string, text: string): boolean {
  return name.toLowerCase().endsWith(".svg") || text.slice(0, 2048).trim().toLowerCase().includes("<svg")
}

export async function loadFile(file: File, params: RasterParams): Promise<ContourBundle> {
  if (file.size > 8 * 1024 * 1024) throw new Error("El archivo supera los 8 MB.")
  const sampleCount = Math.max(64, Math.min(2048, params.sampleCount))
  const head = await file.slice(0, 2048).text()
  if (looksLikeSvg(file.name, head)) {
    const markup = await file.text()
    const extracted = extractSvgContours(markup, sampleCount)
    const first = extracted.contours[0]?.points ?? []
    const size = span(first)
    return {
      source: "svg",
      width: size.width,
      height: size.height,
      preview: null,
      warnings: [
        "Un path con varios M se parte. Los arcos A se muestrean con el navegador.",
        ...extracted.warnings,
      ],
      contours: extracted.contours,
    }
  }
  const raster = await extractRaster(file, { ...params, sampleCount })
  return {
    source: "raster",
    width: raster.width,
    height: raster.height,
    preview: raster.preview,
    warnings: raster.warnings,
    contours: raster.contours,
  }
}

export function analyzeContour(points: Point[], closed: boolean, center: boolean): FourierResult {
  if (points.length > 2048) throw new Error("Como máximo 2048 muestras. Remuestreá el contorno.")
  const spectrum = analyzeSourcePoints(points, center)
  const warnings: string[] = []
  if (!closed) {
    warnings.push("El contorno está abierto. Igual se trata como un período, así que la curva reconstruida cierra el hueco.")
  }
  warnings.push(
    center
      ? "Centrado activo: el término de continua es la posición y los demás coeficientes describen la forma."
      : "Centrado apagado: el primer círculo, de frecuencia 0, lleva la traslación.",
  )
  return {
    sample_count: spectrum.sampleCount,
    closed,
    centered: spectrum.centered,
    offset: [spectrum.offsetX, spectrum.offsetY],
    coefficients: spectrum.coefficients,
    x_coefficients: spectrum.xCoefficients,
    y_coefficients: spectrum.yCoefficients,
    warnings,
  }
}

export async function loadMetrics(body: {
  points: Point[]
  closed: boolean
  center: boolean
  strategy: Strategy
  component_count: number
  number_format: NumberFormat
  quant_step: number
}): Promise<Metrics> {
  return measureCompression(body)
}

function csvLine(values: Array<string | number>): string {
  return values.join(",")
}

function coefficientCsv(points: Point[], center: boolean): string {
  const spectrum = analyzeSourcePoints(points, center)
  const lines = ["frequency,real,imag,amplitude,phase"]
  for (const coefficient of spectrum.coefficients) {
    lines.push(csvLine([
      coefficient.frequency,
      coefficient.real.toPrecision(12),
      coefficient.imag.toPrecision(12),
      coefficient.amplitude.toPrecision(12),
      coefficient.phase.toPrecision(12),
    ]))
  }
  return lines.join("\n") + "\n"
}

function coefficientJson(points: Point[], closed: boolean, center: boolean): string {
  const spectrum = analyzeContour(points, closed, center)
  return JSON.stringify(
    {
      sample_count: spectrum.sample_count,
      centered: spectrum.centered,
      offset: spectrum.offset,
      convention: "z = x - i y, con y de imagen hacia abajo; c_k = FFT(z) / N",
      coefficients: spectrum.coefficients,
      x_coefficients: spectrum.x_coefficients,
      y_coefficients: spectrum.y_coefficients,
    },
    null,
    2,
  )
}

function metricsCsv(request: ExportRequest): string {
  const metrics = measureCompression(request)
  const lines = ["section,key,value"]
  const summary: Record<string, string | number | null> = {
    sample_count: metrics.sample_count,
    component_count: metrics.component_count,
    stored_coefficients: metrics.stored_coefficients,
    number_format: metrics.number_format,
    quant_step: metrics.quant_step,
    raw_points_bytes: metrics.raw_points_bytes,
    raw_coefficients_bytes: metrics.raw_coefficients_bytes,
    encoded_bytes: metrics.encoded_bytes,
    ratio_vs_points: metrics.ratio_vs_points,
    ratio_vs_coefficients: metrics.ratio_vs_coefficients,
    rmse: metrics.rmse,
    max_error: metrics.max_error,
    hausdorff: metrics.hausdorff,
  }
  for (const [key, value] of Object.entries(summary)) lines.push(["summary", key, value ?? ""].join(","))
  lines.push("")
  lines.push("component_count,stored_coefficients,encoded_bytes,rmse,max_error,hausdorff")
  for (const row of metrics.curve) {
    lines.push([
      row.component_count,
      row.stored_coefficients,
      row.encoded_bytes,
      row.rmse.toPrecision(12),
      row.max_error.toPrecision(12),
      row.hausdorff.toPrecision(12),
    ].join(","))
  }
  return lines.join("\n") + "\n"
}

function textBlob(text: string, type: string): Blob {
  return new Blob([text], { type })
}

export async function exportArtifact(kind: string, request: ExportRequest): Promise<{ blob: Blob; filename: string }> {
  if (kind === "svg") {
    const spectrum = analyzeSourcePoints(request.points, request.center)
    const selected = selectCoefficients(spectrum.coefficients, request.strategy, request.component_count)
    const curve = reconstructedCurve(selected, spectrum.sampleCount, spectrum.offsetX, spectrum.offsetY)
    return {
      blob: textBlob(renderSvg(request.points, curve, request.closed, request.appearance.original, request.appearance.reconstruction), "image/svg+xml"),
      filename: "reconstruccion.svg",
    }
  }
  if (kind === "png") return { blob: await renderPng(request), filename: "reconstruccion.png" }
  if (kind === "gif") return { blob: renderGif(request), filename: "epiciclos.gif" }
  if (kind === "webm") return { blob: await renderWebm(request), filename: "epiciclos.webm" }
  if (kind === "csv") return { blob: textBlob(coefficientCsv(request.points, request.center), "text/csv"), filename: "coeficientes.csv" }
  if (kind === "json") {
    return {
      blob: textBlob(coefficientJson(request.points, request.closed, request.center), "application/json"),
      filename: "coeficientes.json",
    }
  }
  if (kind === "flc1") {
    const spectrum = analyzeSourcePoints(request.points, request.center)
    const selected = selectCoefficients(spectrum.coefficients, request.strategy, request.component_count)
    const bytes = encodeFlc1(
      selected,
      spectrum.sampleCount,
      spectrum.offsetX,
      spectrum.offsetY,
      request.closed,
      spectrum.centered,
      request.number_format,
      request.quant_step,
    )
    const copy = new ArrayBuffer(bytes.byteLength)
    new Uint8Array(copy).set(bytes)
    return { blob: new Blob([copy]), filename: "contorno.flc1" }
  }
  if (kind === "metrics") return { blob: textBlob(metricsCsv(request), "text/csv"), filename: "metricas.csv" }
  throw new Error("Exportación no reconocida.")
}

export function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export async function importFlc1File(file: File): Promise<{
  points: Point[]
  closed: boolean
  centered: boolean
  note: string
  sample_count: number
}> {
  const document = decodeFlc1(new Uint8Array(await file.arrayBuffer()))
  return {
    points: reconstructDocument(document),
    closed: document.closed,
    centered: document.centered,
    sample_count: document.sampleCount,
    note: "Estos puntos salen de los coeficientes guardados, no del contorno original. El archivo no conserva el mapa de bits ni las muestras que se descartaron.",
  }
}

export async function loadBasis(): Promise<{ patterns: number[][][] }> {
  return { patterns: basisPatterns() }
}

export function loadJpeg(file: File) {
  return storeJpeg(file)
}

export async function reconstructJpeg(body: {
  image_id: string
  quality: number
  cutoff: number
  grayscale: boolean
  block_row: number
  block_col: number
  channel: number
  edits: BlockEdit[]
}): Promise<JpegResult> {
  return reconstructStored(body)
}
