import hummingbirdSvg from "../assets/hummingbird.svg?raw"
import type { Point } from "./resample.ts"
import { resampleContour } from "./resample.ts"
import { extractSvgContours } from "./svg.ts"

const CENTER = 200
const RADIUS = 120

const NOTES: Record<string, string> = {
  circle: "Un círculo antihorario tiene un solo coeficiente, el de frecuencia +1.",
  square: "Las esquinas decaen lento. Hacen falta muchos armónicos y el corte se ve en los vértices.",
  star: "La polilínea se cruza. El parámetro sigue el trazo, no el borde exterior de la figura.",
  heart: "Curva clásica, suave. Con pocas decenas de coeficientes ya se reconoce.",
  hummingbird: "Silueta cerrada de pocos tramos. Comparala con el cuadrado: acá el espectro cae más rápido.",
}

export function presetNote(name: string): string {
  return NOTES[name] ?? ""
}

export function presetPoints(name: string, sampleCount: number): { points: Point[]; closed: boolean; note: string } {
  if (!(name in NOTES)) throw new Error(`No existe el ejemplo '${name}'.`)
  if (name === "circle") return { points: circle(sampleCount), closed: true, note: NOTES.circle }
  if (name === "square") return { points: square(sampleCount), closed: true, note: NOTES.square }
  if (name === "star") return { points: star(sampleCount), closed: true, note: NOTES.star }
  if (name === "heart") return { points: heart(sampleCount), closed: true, note: NOTES.heart }
  return { points: hummingbird(sampleCount), closed: true, note: NOTES.hummingbird }
}

function circle(sampleCount: number): Point[] {
  const points: Point[] = []
  for (let index = 0; index < sampleCount; index += 1) {
    const theta = (2 * Math.PI * index) / sampleCount
    points.push([CENTER + RADIUS * Math.cos(theta), CENTER - RADIUS * Math.sin(theta)])
  }
  return points
}

function square(sampleCount: number): Point[] {
  const half = RADIUS
  return resampleContour(
    [
      [CENTER - half, CENTER - half],
      [CENTER - half, CENTER + half],
      [CENTER + half, CENTER + half],
      [CENTER + half, CENTER - half],
    ],
    sampleCount,
    true,
  )
}

function star(sampleCount: number): Point[] {
  const vertices: Point[] = []
  for (let index = 0; index < 5; index += 1) {
    const theta = Math.PI / 2 + index * ((4 * Math.PI) / 5)
    vertices.push([CENTER + RADIUS * Math.cos(theta), CENTER - RADIUS * Math.sin(theta)])
  }
  return resampleContour(vertices, sampleCount, true)
}

function heart(sampleCount: number): Point[] {
  const raw: Point[] = []
  for (let index = 0; index < 4000; index += 1) {
    const t = (2 * Math.PI * index) / 4000
    const x = 16 * Math.sin(t) ** 3
    const yUp = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)
    raw.push([x, -yUp])
  }
  let meanX = 0
  let meanY = 0
  for (const [x, y] of raw) {
    meanX += x
    meanY += y
  }
  meanX /= raw.length
  meanY /= raw.length
  let span = 1
  for (const [x, y] of raw) span = Math.max(span, Math.abs(x - meanX), Math.abs(y - meanY))
  const fitted = raw.map(([x, y]) => [
    CENTER + ((x - meanX) * RADIUS) / span,
    CENTER + ((y - meanY) * RADIUS) / span,
  ] as Point)
  return resampleContour(fitted, sampleCount, true)
}

function hummingbird(sampleCount: number): Point[] {
  const extracted = extractSvgContours(hummingbirdSvg, sampleCount)
  const closed = extracted.contours.filter((contour) => contour.closed)
  const pool = closed.length ? closed : extracted.contours
  if (!pool.length) throw new Error("La silueta del colibrí no produjo un contorno.")
  const best = pool.reduce((winner, contour) => (contour.perimeter > winner.perimeter ? contour : winner))
  return best.points
}
