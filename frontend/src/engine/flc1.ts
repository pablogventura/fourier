import type { Coefficient } from "../math/series.ts"
import { reconstructSource } from "../math/series.ts"

const MAGIC = [0x46, 0x4c, 0x43, 0x31]
const HEADER_BYTES = 56
const FLAG_CLOSED = 1
const FLAG_QUANTIZED = 2
const FLAG_FLOAT64 = 4
const FLAG_CENTERED = 8

export type NumberFormat = "float64" | "float32" | "int16"

export type Flc1Document = {
  sampleCount: number
  closed: boolean
  centered: boolean
  numberFormat: NumberFormat
  quantStep: number
  offsetX: number
  offsetY: number
  scaleX: number
  scaleY: number
  coefficients: Coefficient[]
}

function quantize(value: number, step: number): number {
  return Math.max(-32768, Math.min(32767, Math.round(value / step)))
}

export function encodeFlc1(
  coefficients: Coefficient[],
  sampleCount: number,
  offsetX: number,
  offsetY: number,
  closed: boolean,
  centered: boolean,
  numberFormat: NumberFormat,
  quantStep: number,
  scaleX = 1,
  scaleY = 1,
): Uint8Array {
  if (numberFormat === "int16" && quantStep <= 0) throw new Error("El paso de cuantización tiene que ser positivo.")
  let flags = 0
  if (closed) flags |= FLAG_CLOSED
  if (centered) flags |= FLAG_CENTERED
  let storedStep = 0
  if (numberFormat === "int16") {
    flags |= FLAG_QUANTIZED
    storedStep = quantStep
  } else if (numberFormat === "float64") {
    flags |= FLAG_FLOAT64
  }
  const recordBytes = numberFormat === "int16" ? 8 : numberFormat === "float64" ? 20 : 12
  const payload = new Uint8Array(HEADER_BYTES + coefficients.length * recordBytes)
  const view = new DataView(payload.buffer)
  payload.set(MAGIC, 0)
  view.setUint16(4, 1, true)
  view.setUint16(6, flags, true)
  view.setUint32(8, sampleCount, true)
  view.setUint32(12, coefficients.length, true)
  view.setFloat64(16, offsetX, true)
  view.setFloat64(24, offsetY, true)
  view.setFloat64(32, scaleX, true)
  view.setFloat64(40, scaleY, true)
  view.setFloat64(48, storedStep, true)
  let cursor = HEADER_BYTES
  for (const coefficient of coefficients) {
    view.setInt32(cursor, coefficient.frequency, true)
    cursor += 4
    if (numberFormat === "int16") {
      view.setInt16(cursor, quantize(coefficient.real, quantStep), true)
      view.setInt16(cursor + 2, quantize(coefficient.imag, quantStep), true)
      cursor += 4
    } else if (numberFormat === "float64") {
      view.setFloat64(cursor, coefficient.real, true)
      view.setFloat64(cursor + 8, coefficient.imag, true)
      cursor += 16
    } else {
      view.setFloat32(cursor, coefficient.real, true)
      view.setFloat32(cursor + 4, coefficient.imag, true)
      cursor += 8
    }
  }
  return payload
}

export function decodeFlc1(payload: Uint8Array): Flc1Document {
  if (payload.length < HEADER_BYTES) throw new Error("El archivo es más corto que la cabecera FLC1.")
  if (payload[0] !== MAGIC[0] || payload[1] !== MAGIC[1] || payload[2] !== MAGIC[2] || payload[3] !== MAGIC[3]) {
    throw new Error("El archivo no empieza con la marca FLC1.")
  }
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength)
  const version = view.getUint16(4, true)
  if (version !== 1) throw new Error(`Versión de FLC1 no reconocida: ${version}.`)
  const flags = view.getUint16(6, true)
  const sampleCount = view.getUint32(8, true)
  const count = view.getUint32(12, true)
  const offsetX = view.getFloat64(16, true)
  const offsetY = view.getFloat64(24, true)
  const scaleX = view.getFloat64(32, true)
  const scaleY = view.getFloat64(40, true)
  const quantStep = view.getFloat64(48, true)
  const quantized = (flags & FLAG_QUANTIZED) !== 0
  const useFloat64 = (flags & FLAG_FLOAT64) !== 0
  const numberFormat: NumberFormat = quantized ? "int16" : useFloat64 ? "float64" : "float32"
  const recordBytes = quantized ? 8 : useFloat64 ? 20 : 12
  if (payload.length < HEADER_BYTES + count * recordBytes) throw new Error("El archivo FLC1 está truncado.")
  const coefficients: Coefficient[] = []
  let cursor = HEADER_BYTES
  for (let index = 0; index < count; index += 1) {
    const frequency = view.getInt32(cursor, true)
    cursor += 4
    let real: number
    let imag: number
    if (quantized) {
      real = view.getInt16(cursor, true) * quantStep
      imag = view.getInt16(cursor + 2, true) * quantStep
      cursor += 4
    } else if (useFloat64) {
      real = view.getFloat64(cursor, true)
      imag = view.getFloat64(cursor + 8, true)
      cursor += 16
    } else {
      real = view.getFloat32(cursor, true)
      imag = view.getFloat32(cursor + 4, true)
      cursor += 8
    }
    coefficients.push({
      frequency,
      real,
      imag,
      amplitude: Math.hypot(real, imag),
      phase: Math.atan2(imag, real),
    })
  }
  return {
    sampleCount,
    closed: (flags & FLAG_CLOSED) !== 0,
    centered: (flags & FLAG_CENTERED) !== 0,
    numberFormat,
    quantStep,
    offsetX,
    offsetY,
    scaleX,
    scaleY,
    coefficients,
  }
}

export function reconstructDocument(document: Flc1Document): Array<[number, number]> {
  const tips = reconstructSource(document.coefficients, document.sampleCount, 0, 0)
  return tips.map((point) => [document.offsetX + document.scaleX * point.x, document.offsetY + document.scaleY * point.y])
}
