import type { Contour } from "../types.ts"
import { perimeter, resampleContour, type Point } from "./resample.ts"
import { makeContour } from "./svg.ts"

export type RasterParams = {
  threshold: number
  blurSigma: number
  minLength: number
  method: "binary" | "canny"
  polarity: "auto" | "bright" | "dark"
  sampleCount: number
}

const DIRS: Array<[number, number]> = [
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 1],
  [-1, 0],
  [-1, -1],
  [0, -1],
  [1, -1],
]

function reflectIndex(index: number, length: number): number {
  if (length <= 1) return 0
  const period = 2 * (length - 1)
  let mapped = index % period
  if (mapped < 0) mapped += period
  if (mapped >= length) mapped = period - mapped
  return mapped
}

function gaussianBlur(source: Float64Array, width: number, height: number, sigma: number): Float64Array {
  if (sigma < 0.05) return source
  const radius = Math.min(24, Math.ceil(sigma * 3))
  const kernel = new Float64Array(radius * 2 + 1)
  let weight = 0
  for (let offset = -radius; offset <= radius; offset += 1) {
    const value = Math.exp(-(offset * offset) / (2 * sigma * sigma))
    kernel[offset + radius] = value
    weight += value
  }
  for (let index = 0; index < kernel.length; index += 1) kernel[index] /= weight
  const horizontal = new Float64Array(width * height)
  const output = new Float64Array(width * height)
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let sum = 0
      for (let offset = -radius; offset <= radius; offset += 1) {
        const xx = reflectIndex(x + offset, width)
        sum += source[y * width + xx] * kernel[offset + radius]
      }
      horizontal[y * width + x] = sum
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let sum = 0
      for (let offset = -radius; offset <= radius; offset += 1) {
        const yy = reflectIndex(y + offset, height)
        sum += horizontal[yy * width + x] * kernel[offset + radius]
      }
      output[y * width + x] = sum
    }
  }
  return output
}

function components(mask: Uint8Array, width: number, height: number): number[][] {
  const labels = new Int32Array(width * height)
  const groups: number[][] = []
  const stack: number[] = []
  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || labels[start]) continue
    const pixels: number[] = []
    labels[start] = groups.length + 1
    stack.push(start)
    while (stack.length) {
      const pixel = stack.pop()!
      pixels.push(pixel)
      const x = pixel % width
      const y = (pixel / width) | 0
      if (x + 1 < width && mask[pixel + 1] && !labels[pixel + 1]) {
        labels[pixel + 1] = labels[pixel]
        stack.push(pixel + 1)
      }
      if (x > 0 && mask[pixel - 1] && !labels[pixel - 1]) {
        labels[pixel - 1] = labels[pixel]
        stack.push(pixel - 1)
      }
      if (y + 1 < height && mask[pixel + width] && !labels[pixel + width]) {
        labels[pixel + width] = labels[pixel]
        stack.push(pixel + width)
      }
      if (y > 0 && mask[pixel - width] && !labels[pixel - width]) {
        labels[pixel - width] = labels[pixel]
        stack.push(pixel - width)
      }
    }
    groups.push(pixels)
  }
  return groups
}

function trace(width: number, height: number, pixels: number[]): Point[] {
  const member = new Uint8Array(width * height)
  let start = pixels[0]
  for (const pixel of pixels) {
    member[pixel] = 1
    const y = (pixel / width) | 0
    const x = pixel % width
    const sy = (start / width) | 0
    const sx = start % width
    if (y < sy || (y === sy && x < sx)) start = pixel
  }
  let x = start % width
  let y = (start / width) | 0
  let backX = x - 1
  let backY = y
  const points: Point[] = []
  let firstNextX = -2
  let firstNextY = -2
  const limit = Math.max(64, pixels.length * 8)
  for (let guard = 0; guard < limit; guard += 1) {
    points.push([x + 0.5, y + 0.5])
    let startDir = 0
    for (let dir = 0; dir < 8; dir += 1) {
      if (x + DIRS[dir][0] === backX && y + DIRS[dir][1] === backY) {
        startDir = (dir + 1) % 8
        break
      }
    }
    let nextX = x
    let nextY = y
    let found = false
    for (let step = 0; step < 8; step += 1) {
      const dir = (startDir + step) % 8
      const cx = x + DIRS[dir][0]
      const cy = y + DIRS[dir][1]
      if (cx < 0 || cy < 0 || cx >= width || cy >= height || !member[cy * width + cx]) continue
      const previous = (dir + 7) % 8
      backX = x + DIRS[previous][0]
      backY = y + DIRS[previous][1]
      nextX = cx
      nextY = cy
      found = true
      break
    }
    if (!found) break
    if (points.length > 2 && x === start % width && y === ((start / width) | 0) && nextX === firstNextX && nextY === firstNextY) {
      points.pop()
      break
    }
    if (points.length === 1) {
      firstNextX = nextX
      firstNextY = nextY
    }
    x = nextX
    y = nextY
  }
  return points
}

function applyPolarity(mask: Uint8Array, width: number, height: number, polarity: RasterParams["polarity"]): string | null {
  if (polarity === "dark") {
    for (let index = 0; index < mask.length; index += 1) mask[index] = mask[index] ? 0 : 1
    return "Polaridad oscura: la figura es lo que está debajo del umbral."
  }
  if (polarity === "bright") return null
  let sum = 0
  let count = 0
  for (let x = 0; x < width; x += 1) {
    sum += mask[x] + mask[(height - 1) * width + x]
    count += 2
  }
  for (let y = 0; y < height; y += 1) {
    sum += mask[y * width] + mask[y * width + width - 1]
    count += 2
  }
  if (sum / count > 0.5) {
    for (let index = 0; index < mask.length; index += 1) mask[index] = mask[index] ? 0 : 1
    return "Se invirtió la máscara: el borde de la imagen estaba del lado de la figura."
  }
  return null
}

function canny(gray: Float64Array, width: number, height: number, threshold: number): Uint8Array {
  const magnitude = new Float64Array(width * height)
  const strong = new Uint8Array(width * height)
  const high = threshold
  const low = threshold * 0.4
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x
      const gx =
        -gray[index - width - 1] + gray[index - width + 1]
        - 2 * gray[index - 1] + 2 * gray[index + 1]
        - gray[index + width - 1] + gray[index + width + 1]
      const gy =
        -gray[index - width - 1] - 2 * gray[index - width] - gray[index - width + 1]
        + gray[index + width - 1] + 2 * gray[index + width] + gray[index + width + 1]
      const value = Math.hypot(gx, gy)
      magnitude[index] = value
      if (value >= high) strong[index] = 2
      else if (value >= low) strong[index] = 1
    }
  }
  const mask = new Uint8Array(width * height)
  let changed = true
  while (changed) {
    changed = false
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const index = y * width + x
        if (strong[index] !== 1) continue
        for (const [dx, dy] of DIRS) {
          if (strong[index + dy * width + dx] === 2) {
            strong[index] = 2
            changed = true
            break
          }
        }
      }
    }
  }
  for (let index = 0; index < strong.length; index += 1) if (strong[index] === 2) mask[index] = 1
  return mask
}

export async function extractRaster(file: File, params: RasterParams): Promise<{
  contours: Contour[]
  warnings: string[]
  preview: string
  width: number
  height: number
}> {
  if (file.size > 8 * 1024 * 1024) throw new Error("El archivo supera los 8 MB.")
  const bitmap = await createImageBitmap(file)
  const scale = Math.min(1, 1600 / Math.max(bitmap.width, bitmap.height))
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
  const preview = canvas.toDataURL("image/jpeg", 0.85)
  let usefulAlpha = false
  let transparent = 0
  let opaque = 0
  for (let index = 3; index < image.data.length; index += 4) {
    if (image.data[index] < 250) transparent += 1
    if (image.data[index] > 5) opaque += 1
  }
  usefulAlpha = transparent > 0 && opaque > 0
  const gray = new Float64Array(width * height)
  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 4
    gray[index] = usefulAlpha
      ? image.data[offset + 3]
      : 0.299 * image.data[offset] + 0.587 * image.data[offset + 1] + 0.114 * image.data[offset + 2]
  }
  const blurred = gaussianBlur(gray, width, height, params.blurSigma)
  const warnings: string[] = []
  if (usefulAlpha) {
    warnings.push("Se usó el canal alfa como máscara, así el fondo transparente no genera un rectángulo.")
  }
  let mask: Uint8Array
  if (params.method === "canny") {
    mask = canny(blurred, width, height, params.threshold)
    warnings.push("Canny marca bordes. El contorno recorre el borde como un lazo fino, no rellena la figura.")
  } else {
    mask = new Uint8Array(width * height)
    const threshold = Math.max(0, Math.min(255, params.threshold))
    for (let index = 0; index < mask.length; index += 1) mask[index] = blurred[index] > threshold ? 1 : 0
    const note = applyPolarity(mask, width, height, params.polarity)
    if (note) warnings.push(note)
  }
  const foreground = components(mask, width, height)
  const holes = components(
    Uint8Array.from(mask, (value) => (value ? 0 : 1)),
    width,
    height,
  ).filter((pixels) => pixels.some((pixel) => {
    const x = pixel % width
    const y = (pixel / width) | 0
    return x > 0 && y > 0 && x < width - 1 && y < height - 1
  }) && pixels.every((pixel) => {
    const x = pixel % width
    const y = (pixel / width) | 0
    return x > 0 && y > 0 && x < width - 1 && y < height - 1
  }))
  const traced: Array<{ points: Point[]; hole: boolean }> = []
  for (const pixels of foreground) traced.push({ points: trace(width, height, pixels), hole: false })
  for (const pixels of holes) traced.push({ points: trace(width, height, pixels), hole: true })
  const contours: Contour[] = []
  for (const item of traced) {
    if (item.points.length < 4) continue
    let length = 0
    try {
      length = perimeter(item.points, true)
    } catch {
      continue
    }
    if (length < params.minLength) continue
    let sampled: Point[]
    try {
      sampled = resampleContour(item.points, params.sampleCount, true)
    } catch {
      continue
    }
    const contour = makeContour(0, sampled, true, item.hole, item.hole ? "agujero" : "contorno")
    contour.perimeter = length
    contours.push(contour)
  }
  contours.sort((left, right) => right.perimeter - left.perimeter)
  contours.forEach((contour, index) => {
    contour.id = index
  })
  if (!contours.length) {
    warnings.push(
      foreground.length
        ? "Los contornos encontrados son más cortos que la longitud mínima."
        : "No se detectó ningún contorno. Probá otro umbral, menos desenfoque o el método de Canny.",
    )
  } else if (contours.length > 1) {
    warnings.push("Hay varios contornos. Elegí uno solo: una serie de Fourier describe una curva, no la unión de piezas sueltas.")
  }
  return { contours, warnings, preview, width, height }
}
