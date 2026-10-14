import { describe, expect, it } from "vitest"
import golden from "./golden.json" with { type: "json" }
import { mathTips, selectCoefficients, snappedCount, type Coefficient } from "./series.ts"

describe("serie de Fourier en el canvas", () => {
  it("reproduce los fasores de NumPy en el tono doble", () => {
    const tips = mathTips(golden.twoTone.coefficients, golden.twoTone.t)
    expect(tips).toHaveLength(golden.twoTone.tips.length)
    tips.forEach((tip, index) => {
      expect(tip.x).toBeCloseTo(golden.twoTone.tips[index][0], 6)
      expect(tip.y).toBeCloseTo(golden.twoTone.tips[index][1], 6)
    })
  })

  it("reproduce un fasor con fase distinta de cero", () => {
    const tips = mathTips(golden.phased.coefficients, golden.phased.t)
    const tip = tips[0]
    expect(tip.x).toBeCloseTo(golden.phased.tips[0][0], 6)
    expect(tip.y).toBeCloseTo(golden.phased.tips[0][1], 6)
  })

  it("ajusta el truncamiento simétrico a una cantidad impar", () => {
    expect(snappedCount("symmetric", 4, 32)).toBe(3)
    const coefficients: Coefficient[] = [-2, -1, 0, 1, 2, 10].map((frequency) => ({
      frequency,
      real: frequency === 10 ? 0.4 : 0.1,
      imag: 0,
      amplitude: frequency === 10 ? 0.4 : 0.1,
      phase: 0,
    }))
    const symmetric = selectCoefficients(coefficients, "symmetric", 5).map((item) => item.frequency)
    const magnitude = selectCoefficients(coefficients, "magnitude", 2).map((item) => item.frequency)
    expect(symmetric).toEqual([0, 1, -1, 2, -2])
    expect(magnitude).toEqual([0, 10])
  })
})
