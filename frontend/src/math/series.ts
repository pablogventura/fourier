export type Coefficient = {
  frequency: number
  real: number
  imag: number
  amplitude: number
  phase: number
}

export type Strategy = "symmetric" | "magnitude"

export function snappedCount(strategy: Strategy, componentCount: number, available: number): number {
  const count = Math.max(1, Math.min(componentCount, available, 500))
  if (strategy === "symmetric" && count % 2 === 0) return Math.max(1, count - 1)
  return count
}

function visualRank(frequency: number): [number, number] {
  return [Math.abs(frequency), frequency >= 0 ? 0 : 1]
}

export function selectCoefficients(
  coefficients: Coefficient[],
  strategy: Strategy,
  componentCount: number,
): Coefficient[] {
  const count = snappedCount(strategy, componentCount, coefficients.length)
  let chosen: Coefficient[]
  if (strategy === "magnitude") {
    chosen = [...coefficients]
      .sort((a, b) => {
        if (b.amplitude !== a.amplitude) return b.amplitude - a.amplitude
        const [aa, as] = visualRank(a.frequency)
        const [ba, bs] = visualRank(b.frequency)
        return aa - ba || as - bs
      })
      .slice(0, count)
  } else {
    const kMax = (count - 1) / 2
    chosen = coefficients.filter((coefficient) => Math.abs(coefficient.frequency) <= kMax)
  }
  return chosen.sort((a, b) => {
    const [aa, as] = visualRank(a.frequency)
    const [ba, bs] = visualRank(b.frequency)
    return aa - ba || as - bs
  })
}

export function rotate(coefficient: Coefficient, t: number): { re: number; im: number } {
  const theta = 2 * Math.PI * coefficient.frequency * t
  const cos = Math.cos(theta)
  const sin = Math.sin(theta)
  return {
    re: coefficient.real * cos - coefficient.imag * sin,
    im: coefficient.real * sin + coefficient.imag * cos,
  }
}

export function mathTips(coefficients: Coefficient[], t: number): Array<{ x: number; y: number }> {
  let x = 0
  let y = 0
  const tips: Array<{ x: number; y: number }> = []
  for (const coefficient of coefficients) {
    const step = rotate(coefficient, t)
    x += step.re
    y += step.im
    tips.push({ x, y })
  }
  return tips
}

export type Circle = {
  cx: number
  cy: number
  x: number
  y: number
  radius: number
  frequency: number
}

export function sourceChain(
  coefficients: Coefficient[],
  t: number,
  offsetX: number,
  offsetY: number,
): Circle[] {
  let x = offsetX
  let y = offsetY
  const circles: Circle[] = []
  for (const coefficient of coefficients) {
    const step = rotate(coefficient, t)
    const nextX = x + step.re
    const nextY = y - step.im
    circles.push({
      cx: x,
      cy: y,
      x: nextX,
      y: nextY,
      radius: Math.hypot(coefficient.real, coefficient.imag),
      frequency: coefficient.frequency,
    })
    x = nextX
    y = nextY
  }
  return circles
}

export function axisChain(
  coefficients: Coefficient[],
  t: number,
  anchorX: number,
  anchorY: number,
  axis: "x" | "y",
): Circle[] {
  let x = anchorX
  let y = anchorY
  const circles: Circle[] = []
  for (const coefficient of coefficients) {
    const step = rotate(coefficient, t)
    const dx = axis === "x" ? step.re : step.im
    const dy = axis === "x" ? -step.im : step.re
    const nextX = x + dx
    const nextY = y + dy
    circles.push({
      cx: x,
      cy: y,
      x: nextX,
      y: nextY,
      radius: Math.hypot(coefficient.real, coefficient.imag),
      frequency: coefficient.frequency,
    })
    x = nextX
    y = nextY
  }
  return circles
}

export function realValue(coefficients: Coefficient[], t: number): number {
  let value = 0
  for (const coefficient of coefficients) value += rotate(coefficient, t).re
  return value
}

export function reconstructSource(
  coefficients: Coefficient[],
  sampleCount: number,
  offsetX: number,
  offsetY: number,
): Array<{ x: number; y: number }> {
  const points: Array<{ x: number; y: number }> = []
  for (let index = 0; index < sampleCount; index += 1) {
    const tips = mathTips(coefficients, index / sampleCount)
    const tip = tips.at(-1) ?? { x: 0, y: 0 }
    points.push({ x: offsetX + tip.x, y: offsetY - tip.y })
  }
  return points
}
