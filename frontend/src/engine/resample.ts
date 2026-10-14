export type Point = [number, number]

export function dedupePoints(points: Point[], tolerance = 1e-9): Point[] {
  if (points.length === 0) return []
  const kept: Point[] = [points[0]]
  for (let index = 1; index < points.length; index += 1) {
    const [x, y] = points[index]
    const [px, py] = kept[kept.length - 1]
    if (Math.hypot(x - px, y - py) > tolerance) kept.push(points[index])
  }
  return kept
}

export function preparePolyline(points: Point[], closed: boolean): Point[] {
  if (points.length < 2) throw new Error("Un contorno necesita al menos dos puntos.")
  let pts = points
  if (closed && Math.hypot(pts[0][0] - pts[pts.length - 1][0], pts[0][1] - pts[pts.length - 1][1]) <= 1e-9) {
    pts = pts.slice(0, -1)
  }
  pts = dedupePoints(pts)
  const minimum = closed ? 3 : 2
  if (pts.length < minimum) {
    throw new Error("El contorno queda degenerado después de quitar puntos repetidos.")
  }
  return pts
}

export function arcLengthTable(points: Point[], closed: boolean): { vertices: Point[]; cumulative: number[] } {
  const vertices = preparePolyline(points, closed)
  const cumulative = [0]
  if (closed) {
    for (let index = 0; index < vertices.length; index += 1) {
      const next = vertices[(index + 1) % vertices.length]
      const current = vertices[index]
      cumulative.push(cumulative[cumulative.length - 1] + Math.hypot(next[0] - current[0], next[1] - current[1]))
    }
  } else {
    for (let index = 1; index < vertices.length; index += 1) {
      const previous = vertices[index - 1]
      const current = vertices[index]
      cumulative.push(cumulative[cumulative.length - 1] + Math.hypot(current[0] - previous[0], current[1] - previous[1]))
    }
  }
  if (cumulative[cumulative.length - 1] <= 1e-12) throw new Error("La longitud del contorno es nula.")
  return { vertices, cumulative }
}

function searchSortedRight(values: number[], target: number): number {
  let low = 0
  let high = values.length
  while (low < high) {
    const mid = (low + high) >> 1
    if (values[mid] <= target) low = mid + 1
    else high = mid
  }
  return low
}

export function resampleContour(points: Point[], sampleCount: number, closed: boolean): Point[] {
  if (sampleCount < 4) throw new Error("Hacen falta al menos 4 muestras.")
  const { vertices, cumulative } = arcLengthTable(points, closed)
  const total = cumulative[cumulative.length - 1]
  const sampled: Point[] = []
  for (let index = 0; index < sampleCount; index += 1) {
    const target = closed ? (total * index) / sampleCount : (total * index) / (sampleCount - 1)
    let segment: number
    let start: Point
    let end: Point
    let segmentStart: number
    let segmentLength: number
    if (closed) {
      const vertexS = cumulative.slice(0, -1)
      segment = Math.min(vertices.length - 1, Math.max(0, searchSortedRight(vertexS, target) - 1))
      start = vertices[segment]
      end = vertices[(segment + 1) % vertices.length]
      segmentStart = cumulative[segment]
      segmentLength = cumulative[segment + 1] - cumulative[segment]
    } else {
      segment = Math.min(vertices.length - 2, Math.max(0, searchSortedRight(cumulative, target) - 1))
      start = vertices[segment]
      end = vertices[segment + 1]
      segmentStart = cumulative[segment]
      segmentLength = cumulative[segment + 1] - cumulative[segment]
    }
    const local = segmentLength > 1e-15 ? (target - segmentStart) / segmentLength : 0
    sampled.push([start[0] + (end[0] - start[0]) * local, start[1] + (end[1] - start[1]) * local])
  }
  if (!closed) {
    sampled[0] = vertices[0]
    sampled[sampled.length - 1] = vertices[vertices.length - 1]
  }
  return sampled
}

export function perimeter(points: Point[], closed: boolean): number {
  return arcLengthTable(points, closed).cumulative.at(-1) ?? 0
}
