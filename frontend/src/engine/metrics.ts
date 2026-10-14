import type { Coefficient, Strategy } from "../math/series.ts"
import { reconstructSource, selectCoefficients, snappedCount } from "../math/series.ts"
import { analyzeSourcePoints, type SourceSpectrum } from "./dft.ts"
import { decodeFlc1, encodeFlc1, reconstructDocument, type NumberFormat } from "./flc1.ts"
import type { Metrics, RatePoint } from "../types.ts"

const HAUSDORFF_CAP = 256

function distance(left: [number, number], right: [number, number]): number {
  return Math.hypot(left[0] - right[0], left[1] - right[1])
}

function subsample(points: Array<[number, number]>): Array<[number, number]> {
  if (points.length <= HAUSDORFF_CAP) return points
  const sampled: Array<[number, number]> = []
  for (let index = 0; index < HAUSDORFF_CAP; index += 1) {
    const source = Math.round((index * (points.length - 1)) / (HAUSDORFF_CAP - 1))
    sampled.push(points[source])
  }
  return sampled
}

function directedHausdorff(source: Array<[number, number]>, target: Array<[number, number]>): number {
  let worst = 0
  for (const point of source) {
    let best = Infinity
    for (const other of target) best = Math.min(best, distance(point, other))
    worst = Math.max(worst, best)
  }
  return worst
}

function reconstructionErrors(original: Array<[number, number]>, reconstructed: Array<[number, number]>) {
  let sum = 0
  let maxError = 0
  for (let index = 0; index < original.length; index += 1) {
    const error = distance(original[index], reconstructed[index])
    sum += error * error
    maxError = Math.max(maxError, error)
  }
  const left = subsample(original)
  const right = subsample(reconstructed)
  return {
    rmse: Math.sqrt(sum / original.length),
    maxError,
    hausdorff: Math.max(directedHausdorff(left, right), directedHausdorff(right, left)),
  }
}

function evaluateSelected(
  spectrum: SourceSpectrum,
  points: Array<[number, number]>,
  closed: boolean,
  strategy: Strategy,
  componentCount: number,
  numberFormat: NumberFormat,
  quantStep: number,
) {
  const selected = selectCoefficients(spectrum.coefficients, strategy, componentCount)
  const payload = encodeFlc1(
    selected,
    spectrum.sampleCount,
    spectrum.offsetX,
    spectrum.offsetY,
    closed,
    spectrum.centered,
    numberFormat,
    quantStep,
  )
  const reconstructed = reconstructDocument(decodeFlc1(payload))
  const errors = reconstructionErrors(points, reconstructed)
  const kept = snappedCount(strategy, componentCount, spectrum.coefficients.length)
  return {
    componentCount: kept,
    storedCoefficients: selected.length,
    encodedBytes: payload.length,
    quantStep: numberFormat === "int16" ? quantStep : 0,
    rmse: errors.rmse,
    maxError: errors.maxError,
    hausdorff: errors.hausdorff,
    payload,
    selected,
  }
}

function geomspace(start: number, stop: number, count: number): number[] {
  if (count <= 1) return [start]
  const logStart = Math.log(start)
  const logStop = Math.log(stop)
  const values: number[] = []
  for (let index = 0; index < count; index += 1) {
    const t = index / (count - 1)
    values.push(Math.exp(logStart + (logStop - logStart) * t))
  }
  return values
}

export function measureCompression(input: {
  points: Array<[number, number]>
  closed: boolean
  center: boolean
  strategy: Strategy
  component_count: number
  number_format: NumberFormat
  quant_step: number
}): Metrics {
  const spectrum = analyzeSourcePoints(input.points, input.center)
  const focus = evaluateSelected(
    spectrum,
    input.points,
    input.closed,
    input.strategy,
    input.component_count,
    input.number_format,
    input.quant_step,
  )
  const cap = Math.min(spectrum.sampleCount, 500)
  const grid = new Set<number>([1, cap, focus.componentCount])
  if (cap > 1) {
    for (const value of geomspace(1, cap, Math.min(36, cap))) grid.add(Math.max(1, Math.round(value)))
  }
  const counts = [...grid]
    .filter((value) => value >= 1)
    .map((value) => snappedCount(input.strategy, value, spectrum.sampleCount))
    .filter((value, index, all) => all.indexOf(value) === index)
    .sort((left, right) => left - right)
  const curve: RatePoint[] = counts.map((count) => {
    const row = evaluateSelected(
      spectrum,
      input.points,
      input.closed,
      input.strategy,
      count,
      input.number_format,
      input.quant_step,
    )
    return {
      component_count: row.componentCount,
      stored_coefficients: row.storedCoefficients,
      encoded_bytes: row.encodedBytes,
      rmse: row.rmse,
      max_error: row.maxError,
      hausdorff: row.hausdorff,
    }
  })
  const rawPointsBytes = spectrum.sampleCount * 2 * 8
  const rawCoefficientsBytes = spectrum.sampleCount * 2 * 8
  return {
    sample_count: spectrum.sampleCount,
    component_count: focus.componentCount,
    stored_coefficients: focus.storedCoefficients,
    number_format: input.number_format,
    quant_step: focus.quantStep,
    raw_points_bytes: rawPointsBytes,
    raw_coefficients_bytes: rawCoefficientsBytes,
    encoded_bytes: focus.encodedBytes,
    ratio_vs_points: focus.encodedBytes ? rawPointsBytes / focus.encodedBytes : null,
    ratio_vs_coefficients: focus.encodedBytes ? rawCoefficientsBytes / focus.encodedBytes : null,
    rmse: focus.rmse,
    max_error: focus.maxError,
    hausdorff: focus.hausdorff,
    curve,
    note:
      "raw_points_bytes y raw_coefficients_bytes son datos crudos en float64, sin archivo. " +
      "encoded_bytes es el FLC1 real. El cociente no es una tasa JPEG, y el contorno no es la imagen. " +
      "En el navegador, Hausdorff compara como máximo 256 puntos de cada curva.",
  }
}

export function selectedPayload(
  spectrum: SourceSpectrum,
  closed: boolean,
  strategy: Strategy,
  componentCount: number,
  numberFormat: NumberFormat,
  quantStep: number,
): { bytes: Uint8Array; selected: Coefficient[] } {
  const selected = selectCoefficients(spectrum.coefficients, strategy, componentCount)
  return {
    selected,
    bytes: encodeFlc1(
      selected,
      spectrum.sampleCount,
      spectrum.offsetX,
      spectrum.offsetY,
      closed,
      spectrum.centered,
      numberFormat,
      quantStep,
    ),
  }
}

export function reconstructedCurve(coefficients: Coefficient[], sampleCount: number, offsetX: number, offsetY: number) {
  return reconstructSource(coefficients, sampleCount, offsetX, offsetY).map(
    (point) => [point.x, point.y] as [number, number],
  )
}
