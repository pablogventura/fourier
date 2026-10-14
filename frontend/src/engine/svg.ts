import type { Contour } from "../types.ts"
import { perimeter, resampleContour, type Point } from "./resample.ts"

const SVG_NS = "http://www.w3.org/2000/svg"

function userMatrix(node: SVGGraphicsElement, svg: SVGSVGElement): DOMMatrix | null {
  const nodeMatrix = node.getCTM()
  const rootMatrix = svg.getCTM()
  if (!nodeMatrix || !rootMatrix) return nodeMatrix
  return rootMatrix.inverse().multiply(nodeMatrix)
}

function mapPoint(x: number, y: number, matrix: DOMMatrix | null): Point {
  if (!matrix) return [x, y]
  const point = new DOMPoint(x, y).matrixTransform(matrix)
  return [point.x, point.y]
}

function isHidden(element: Element): boolean {
  const style = element.getAttribute("style") ?? ""
  const display = element.getAttribute("display") ?? ""
  const visibility = element.getAttribute("visibility") ?? ""
  return /display\s*:\s*none/.test(style) || display === "none" || visibility === "hidden" || /visibility\s*:\s*hidden/.test(style)
}

function splitSubpaths(description: string): Array<{ d: string; closed: boolean }> {
  return description
    .split(/(?=[Mm])/)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => ({ d: part, closed: /[Zz]/.test(part) }))
}

function sampleGeometry(
  element: SVGGeometryElement,
  svg: SVGSVGElement,
  closed: boolean,
  dense: number,
): Point[] | null {
  const length = element.getTotalLength()
  if (!Number.isFinite(length) || length <= 1e-6) return null
  const matrix = userMatrix(element, svg)
  const count = Math.max(dense, 8)
  const points: Point[] = []
  const steps = closed ? count : count + 1
  for (let index = 0; index < steps; index += 1) {
    const at = element.getPointAtLength(closed ? (length * index) / count : (length * index) / count)
    points.push(mapPoint(at.x, at.y, matrix))
  }
  return points
}

function geometryClosed(element: SVGGeometryElement): boolean {
  const name = element.tagName.toLowerCase().replace(/^.*:/, "")
  if (name === "line" || name === "polyline") return false
  if (name === "path") return false
  return true
}

export function extractSvgContours(markup: string, sampleCount: number): { contours: Contour[]; warnings: string[] } {
  const parsed = new DOMParser().parseFromString(markup, "image/svg+xml")
  if (parsed.querySelector("parsererror")) throw new Error("El SVG no se pudo leer.")
  const source = parsed.querySelector("svg")
  if (!source) throw new Error("El SVG no se pudo leer.")
  const host = document.createElement("div")
  host.style.cssText = "position:fixed;left:-10000px;top:0;visibility:hidden;pointer-events:none"
  host.appendChild(document.importNode(source, true))
  document.body.appendChild(host)
  try {
    const svg = host.querySelector("svg")
    if (!svg) throw new Error("El SVG no se pudo leer.")
    const found: Array<{ points: Point[]; closed: boolean }> = []
    const visit = (element: Element) => {
      if (isHidden(element)) return
      if (element instanceof SVGPathElement) {
        const description = element.getAttribute("d") ?? ""
        const matrix = userMatrix(element, svg)
        for (const chunk of splitSubpaths(description)) {
          const probe = document.createElementNS(SVG_NS, "path")
          probe.setAttribute("d", chunk.d)
          svg.appendChild(probe)
          const sampled = sampleGeometry(probe, svg, chunk.closed, sampleCount * 8)
          probe.remove()
          if (!sampled) continue
          const mapped = matrix ? sampled.map(([x, y]) => mapPoint(x, y, matrix)) : sampled
          found.push({ points: mapped, closed: chunk.closed })
        }
      } else if (element instanceof SVGGeometryElement) {
        const sampled = sampleGeometry(element, svg, geometryClosed(element), sampleCount * 8)
        if (sampled) found.push({ points: sampled, closed: geometryClosed(element) })
      }
      for (const child of element.children) visit(child)
    }
    visit(svg)
    const warnings: string[] = []
    if (!found.length) {
      warnings.push("El SVG no tiene trazados ni formas reconocibles.")
      return { contours: [], warnings }
    }
    const contours: Contour[] = []
    let openCount = 0
    for (const item of found) {
      let sampled: Point[]
      try {
        sampled = resampleContour(item.points, sampleCount, item.closed)
      } catch {
        continue
      }
      if (!item.closed) openCount += 1
      contours.push(makeContour(contours.length, sampled, item.closed, false, item.closed ? "trazado cerrado" : "trazado abierto"))
    }
    if (openCount) {
      warnings.push(
        `${openCount} trazado(s) abierto(s). La serie es periódica: la reconstrucción cierra el salto entre el extremo y el origen, y ahí aparece el fenómeno de Gibbs.`,
      )
    }
    if (contours.length > 1) {
      warnings.push("Cada subtrazado es un contorno independiente. No se concatenan curvas desconectadas.")
    }
    contours.forEach((contour, index) => {
      contour.id = index
    })
    return { contours, warnings }
  } finally {
    host.remove()
  }
}

function bounds(points: Point[]): [number, number, number, number] {
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
  return [minX, minY, maxX, maxY]
}

export function makeContour(id: number, points: Point[], closed: boolean, isHole: boolean, label: string): Contour {
  return {
    id,
    closed,
    is_hole: isHole,
    perimeter: perimeter(points, closed),
    sample_count: points.length,
    points,
    bbox: bounds(points),
    label,
  }
}
