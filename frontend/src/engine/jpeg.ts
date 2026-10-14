import type { JpegResult } from "../types.ts"

const LUMINANCE_Q50 = [
  [16, 11, 10, 16, 24, 40, 51, 61],
  [12, 12, 14, 19, 26, 58, 60, 55],
  [14, 13, 16, 24, 40, 57, 69, 56],
  [14, 17, 22, 29, 51, 87, 80, 62],
  [18, 22, 37, 56, 68, 109, 103, 77],
  [24, 35, 55, 64, 81, 104, 113, 92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103, 99],
]

type StoredImage = { rgb: Uint8Array; width: number; height: number }

const images = new Map<string, StoredImage>()
const order: string[] = []
let nextId = 0

export type BlockEdit = { u: number; v: number; value: number }

const SCALE0 = Math.sqrt(1 / 8)
const SCALE = Math.sqrt(2 / 8)

export function dct8(block: Float64Array): Float64Array {
  const rows = new Float64Array(64)
  const output = new Float64Array(64)
  const line = new Float64Array(8)
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) line[x] = block[y * 8 + x]
    const transformed = dct1(line)
    rows.set(transformed, y * 8)
  }
  for (let x = 0; x < 8; x += 1) {
    for (let y = 0; y < 8; y += 1) line[y] = rows[y * 8 + x]
    const transformed = dct1(line)
    for (let y = 0; y < 8; y += 1) output[y * 8 + x] = transformed[y]
  }
  return output
}

export function idct8(block: Float64Array): Float64Array {
  const rows = new Float64Array(64)
  const output = new Float64Array(64)
  const line = new Float64Array(8)
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) line[x] = block[y * 8 + x]
    const transformed = idct1(line)
    rows.set(transformed, y * 8)
  }
  for (let x = 0; x < 8; x += 1) {
    for (let y = 0; y < 8; y += 1) line[y] = rows[y * 8 + x]
    const transformed = idct1(line)
    for (let y = 0; y < 8; y += 1) output[y * 8 + x] = transformed[y]
  }
  return output
}

function dct1(values: Float64Array): Float64Array {
  const output = new Float64Array(8)
  for (let frequency = 0; frequency < 8; frequency += 1) {
    let sum = 0
    for (let index = 0; index < 8; index += 1) {
      sum += values[index] * Math.cos((Math.PI * frequency * (2 * index + 1)) / 16)
    }
    output[frequency] = sum * (frequency === 0 ? SCALE0 : SCALE)
  }
  return output
}

function idct1(values: Float64Array): Float64Array {
  const output = new Float64Array(8)
  for (let index = 0; index < 8; index += 1) {
    let sum = values[0] * SCALE0
    for (let frequency = 1; frequency < 8; frequency += 1) {
      sum += values[frequency] * SCALE * Math.cos((Math.PI * frequency * (2 * index + 1)) / 16)
    }
    output[index] = sum
  }
  return output
}

export function qualityTable(quality: number): number[][] {
  const level = Math.max(1, Math.min(100, Math.round(quality)))
  const scale = level < 50 ? 5000 / level : 200 - 2 * level
  return LUMINANCE_Q50.map((row) => row.map((value) => Math.max(1, Math.min(255, Math.floor((value * scale + 50) / 100)))))
}

export function basisPatterns(): number[][][] {
  const samples = [0, 1, 2, 3, 4, 5, 6, 7]
  const vectors = samples.map((frequency) => {
    if (frequency === 0) return samples.map(() => 1 / Math.sqrt(8))
    return samples.map((index) => Math.sqrt(2 / 8) * Math.cos((Math.PI * frequency * (2 * index + 1)) / 16))
  })
  const patterns: number[][][] = []
  for (let v = 0; v < 8; v += 1) {
    for (let u = 0; u < 8; u += 1) {
      patterns.push(vectors[v].map((row) => vectors[u].map((column) => row * column)))
    }
  }
  return patterns
}

function reflect(index: number, length: number): number {
  if (length <= 1) return 0
  const period = 2 * (length - 1)
  let mapped = index % period
  if (mapped < 0) mapped += period
  if (mapped >= length) mapped = period - mapped
  return mapped
}

function gaussian11(): Float64Array {
  const kernel = new Float64Array(121)
  let sum = 0
  for (let y = -5; y <= 5; y += 1) {
    for (let x = -5; x <= 5; x += 1) {
      const value = Math.exp(-(x * x + y * y) / (2 * 1.5 * 1.5))
      kernel[(y + 5) * 11 + (x + 5)] = value
      sum += value
    }
  }
  for (let index = 0; index < kernel.length; index += 1) kernel[index] /= sum
  return kernel
}

const GAUSSIAN = gaussian11()

function convolve(source: Float64Array, width: number, height: number): Float64Array {
  const output = new Float64Array(width * height)
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let sum = 0
      let kernel = 0
      for (let ky = -5; ky <= 5; ky += 1) {
        const yy = reflect(y + ky, height)
        for (let kx = -5; kx <= 5; kx += 1) {
          const xx = reflect(x + kx, width)
          sum += source[yy * width + xx] * GAUSSIAN[kernel]
          kernel += 1
        }
      }
      output[y * width + x] = sum
    }
  }
  return output
}

function planeOf(rgb: Uint8Array, width: number, height: number, channel: number): Float64Array {
  const plane = new Float64Array(width * height)
  for (let index = 0; index < plane.length; index += 1) plane[index] = rgb[index * 3 + channel]
  return plane
}

export function ssimPlane(original: Float64Array, reconstructed: Float64Array, width: number, height: number): number {
  const muA = convolve(original, width, height)
  const muB = convolve(reconstructed, width, height)
  const squareA = new Float64Array(original.length)
  const squareB = new Float64Array(original.length)
  const product = new Float64Array(original.length)
  for (let index = 0; index < original.length; index += 1) {
    squareA[index] = original[index] * original[index]
    squareB[index] = reconstructed[index] * reconstructed[index]
    product[index] = original[index] * reconstructed[index]
  }
  const sigmaA = convolve(squareA, width, height)
  const sigmaB = convolve(squareB, width, height)
  const sigmaAB = convolve(product, width, height)
  const c1 = (0.01 * 255) ** 2
  const c2 = (0.03 * 255) ** 2
  let sum = 0
  for (let index = 0; index < original.length; index += 1) {
    const varianceA = sigmaA[index] - muA[index] * muA[index]
    const varianceB = sigmaB[index] - muB[index] * muB[index]
    const covariance = sigmaAB[index] - muA[index] * muB[index]
    const numerator = (2 * muA[index] * muB[index] + c1) * (2 * covariance + c2)
    const denominator = (muA[index] * muA[index] + muB[index] * muB[index] + c1) * (varianceA + varianceB + c2)
    sum += numerator / denominator
  }
  return sum / original.length
}

function psnr(original: Float64Array, reconstructed: Float64Array): number | null {
  let sum = 0
  for (let index = 0; index < original.length; index += 1) {
    const delta = original[index] - reconstructed[index]
    sum += delta * delta
  }
  const mse = sum / original.length
  if (mse <= 1e-12) return null
  return 10 * Math.log10((255 * 255) / mse)
}

function dataUrl(rgb: Uint8Array, width: number, height: number): string {
  const canvas = document.createElement("canvas")
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext("2d")
  if (!context) throw new Error("El navegador no pudo dibujar la imagen.")
  const image = context.createImageData(width, height)
  for (let index = 0; index < width * height; index += 1) {
    image.data[index * 4] = rgb[index * 3]
    image.data[index * 4 + 1] = rgb[index * 3 + 1]
    image.data[index * 4 + 2] = rgb[index * 3 + 2]
    image.data[index * 4 + 3] = 255
  }
  context.putImageData(image, 0, 0)
  return canvas.toDataURL("image/png")
}

function padImage(source: Uint8ClampedArray, sourceWidth: number, sourceHeight: number): StoredImage {
  const padHeight = (8 - (sourceHeight % 8)) % 8
  const padWidth = (8 - (sourceWidth % 8)) % 8
  const width = sourceWidth + padWidth
  const height = sourceHeight + padHeight
  const rgb = new Uint8Array(width * height * 3)
  for (let y = 0; y < height; y += 1) {
    const sy = Math.min(y, sourceHeight - 1)
    for (let x = 0; x < width; x += 1) {
      const sx = Math.min(x, sourceWidth - 1)
      const from = (sy * sourceWidth + sx) * 4
      const to = (y * width + x) * 3
      rgb[to] = source[from]
      rgb[to + 1] = source[from + 1]
      rgb[to + 2] = source[from + 2]
    }
  }
  return { rgb, width, height }
}

export async function storeJpeg(file: File): Promise<{ image_id: string; width: number; height: number; preview: string; note: string }> {
  if (file.size > 8 * 1024 * 1024) throw new Error("El archivo supera los 8 MB.")
  const bitmap = await createImageBitmap(file)
  const scale = Math.min(1, 384 / Math.max(bitmap.width, bitmap.height))
  const width = Math.max(1, Math.round(bitmap.width * scale))
  const height = Math.max(1, Math.round(bitmap.height * scale))
  const canvas = document.createElement("canvas")
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext("2d", { willReadFrequently: true })
  if (!context) throw new Error("El navegador no pudo leer la imagen.")
  context.drawImage(bitmap, 0, 0, width, height)
  bitmap.close()
  const image = context.getImageData(0, 0, width, height)
  const stored = padImage(image.data, width, height)
  const imageId = `img-${nextId}`
  nextId += 1
  images.set(imageId, stored)
  order.push(imageId)
  while (order.length > 3) {
    const oldest = order.shift()
    if (oldest) images.delete(oldest)
  }
  return {
    image_id: imageId,
    width: stored.width,
    height: stored.height,
    preview: dataUrl(stored.rgb, stored.width, stored.height),
    note: "La imagen de trabajo se recorta a un múltiplo de 8 y a 384 px de lado como máximo.",
  }
}

function processPlane(
  plane: Float64Array,
  width: number,
  height: number,
  table: number[][],
  cutoff: number,
  edits: BlockEdit[],
  blockRow: number,
  blockCol: number,
  applyEdits: boolean,
): { restored: Uint8Array; block: { original: number[][]; quantized: number[][]; dequantized: number[][] } | null } {
  const restored = new Uint8Array(width * height)
  const rows = height / 8
  const cols = width / 8
  const rowIndex = Math.max(0, Math.min(rows - 1, blockRow))
  const colIndex = Math.max(0, Math.min(cols - 1, blockCol))
  let selected: { original: number[][]; quantized: number[][]; dequantized: number[][] } | null = null
  const window = new Float64Array(64)
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      for (let y = 0; y < 8; y += 1) {
        for (let x = 0; x < 8; x += 1) window[y * 8 + x] = plane[(row * 8 + y) * width + col * 8 + x] - 128
      }
      const original = dct8(window)
      const dequant = new Float64Array(64)
      const quantized: number[][] = []
      const originalRows: number[][] = []
      const dequantRows: number[][] = []
      for (let v = 0; v < 8; v += 1) {
        const qRow: number[] = []
        const oRow: number[] = []
        const dRow: number[] = []
        for (let u = 0; u < 8; u += 1) {
          const value = original[v * 8 + u]
          const q = Math.round(value / table[v][u])
          let restoredValue = q * table[v][u]
          if (cutoff < 14 && u + v > cutoff) restoredValue = 0
          dequant[v * 8 + u] = restoredValue
          qRow.push(q)
          oRow.push(value)
          dRow.push(restoredValue)
        }
        quantized.push(qRow)
        originalRows.push(oRow)
        dequantRows.push(dRow)
      }
      if (applyEdits && row === rowIndex && col === colIndex) {
        for (const edit of edits) {
          if (edit.u >= 0 && edit.u < 8 && edit.v >= 0 && edit.v < 8) {
            dequant[edit.v * 8 + edit.u] = edit.value
            dequantRows[edit.v][edit.u] = edit.value
          }
        }
      }
      if (row === rowIndex && col === colIndex) {
        selected = { original: originalRows, quantized, dequantized: dequantRows }
      }
      const spatial = idct8(dequant)
      for (let y = 0; y < 8; y += 1) {
        for (let x = 0; x < 8; x += 1) {
          const value = Math.round(spatial[y * 8 + x] + 128)
          restored[(row * 8 + y) * width + col * 8 + x] = Math.max(0, Math.min(255, value))
        }
      }
    }
  }
  return { restored, block: selected }
}

export function reconstructStored(input: {
  image_id: string
  quality: number
  cutoff: number
  grayscale: boolean
  block_row: number
  block_col: number
  channel: number
  edits: BlockEdit[]
}): JpegResult {
  const stored = images.get(input.image_id)
  if (!stored) throw new Error("Esa imagen ya no está en memoria. Volvé a cargarla.")
  const table = qualityTable(input.quality)
  const channel = Math.max(0, Math.min(2, input.channel))
  const { rgb, width, height } = stored
  let originalRgb = rgb
  let reconstructed = new Uint8Array(width * height * 3)
  let metricA: Float64Array
  let metricB: Float64Array
  let block: JpegResult["block"] = null
  if (input.grayscale) {
    const gray = new Float64Array(width * height)
    const grayRgb = new Uint8Array(width * height * 3)
    for (let index = 0; index < gray.length; index += 1) {
      const value = Math.round(0.299 * rgb[index * 3] + 0.587 * rgb[index * 3 + 1] + 0.114 * rgb[index * 3 + 2])
      gray[index] = value
      grayRgb[index * 3] = value
      grayRgb[index * 3 + 1] = value
      grayRgb[index * 3 + 2] = value
    }
    originalRgb = grayRgb
    const processed = processPlane(gray, width, height, table, input.cutoff, input.edits, input.block_row, input.block_col, true)
    for (let index = 0; index < processed.restored.length; index += 1) {
      reconstructed[index * 3] = processed.restored[index]
      reconstructed[index * 3 + 1] = processed.restored[index]
      reconstructed[index * 3 + 2] = processed.restored[index]
    }
    metricA = gray
    metricB = Float64Array.from(processed.restored)
    if (processed.block) {
      block = { row: input.block_row, col: input.block_col, ...processed.block }
    }
  } else {
    let selected: JpegResult["block"] = null
    for (let index = 0; index < 3; index += 1) {
      const plane = planeOf(rgb, width, height, index)
      const processed = processPlane(
        plane,
        width,
        height,
        table,
        input.cutoff,
        input.edits,
        input.block_row,
        input.block_col,
        index === channel,
      )
      for (let pixel = 0; pixel < processed.restored.length; pixel += 1) reconstructed[pixel * 3 + index] = processed.restored[pixel]
      if (index === channel && processed.block) {
        selected = { row: input.block_row, col: input.block_col, ...processed.block }
      }
    }
    block = selected
    metricA = new Float64Array(rgb)
    metricB = new Float64Array(reconstructed)
  }
  const difference = new Uint8Array(width * height * 3)
  for (let index = 0; index < difference.length; index += 1) {
    difference[index] = Math.max(0, Math.min(255, Math.abs(originalRgb[index] - reconstructed[index]) * 4))
  }
  const ssim = input.grayscale
    ? ssimPlane(metricA, metricB, width, height)
    : (ssimPlane(planeOf(originalRgb, width, height, 0), planeOf(reconstructed, width, height, 0), width, height)
      + ssimPlane(planeOf(originalRgb, width, height, 1), planeOf(reconstructed, width, height, 1), width, height)
      + ssimPlane(planeOf(originalRgb, width, height, 2), planeOf(reconstructed, width, height, 2), width, height)) / 3
  return {
    width,
    height,
    blocks_x: width / 8,
    blocks_y: height / 8,
    psnr: input.grayscale ? psnr(metricA, metricB) : psnr(new Float64Array(originalRgb), new Float64Array(reconstructed)),
    ssim,
    block,
    quality_table: table,
    grayscale: input.grayscale,
    channel: input.grayscale ? 0 : channel,
    original: dataUrl(originalRgb, width, height),
    reconstructed: dataUrl(reconstructed, width, height),
    difference: dataUrl(difference, width, height),
    note: "Esto no es un archivo JPEG. Falta el paso a YCbCr, el submuestreo de color y la codificación entrópica. La diferencia está amplificada x4 para que se vea.",
  }
}
