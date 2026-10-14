import { describe, expect, it } from "vitest"
import { analyzeSourcePoints, signedFrequency } from "./dft.ts"

describe("dft del navegador", () => {
  it("etiqueta el bin de Nyquist como +N/2", () => {
    expect(signedFrequency(3, 6)).toBe(3)
    expect([0, 1, 2, 3, 4, 5].map((bin) => signedFrequency(bin, 6))).toEqual([0, 1, 2, 3, -2, -1])
  })

  it("pone un círculo antihorario en la frecuencia +1", () => {
    for (const sampleCount of [6, 64, 192]) {
      const radius = 120
      const points: Array<[number, number]> = []
      for (let index = 0; index < sampleCount; index += 1) {
        const theta = (2 * Math.PI * index) / sampleCount
        points.push([200 + radius * Math.cos(theta), 200 - radius * Math.sin(theta)])
      }
      const spectrum = analyzeSourcePoints(points, true)
      const dominant = spectrum.coefficients.find((item) => item.frequency === 1)
      expect(dominant).toBeTruthy()
      expect(dominant?.real).toBeCloseTo(radius, 6)
      expect(dominant?.imag).toBeCloseTo(0, 6)
      const leftover = spectrum.coefficients
        .filter((item) => item.frequency !== 1)
        .reduce((sum, item) => sum + item.amplitude, 0)
      expect(leftover).toBeLessThan(1e-6)
    }
  })
})
