import { describe, expect, it } from "vitest"
import { dct8, idct8 } from "./jpeg.ts"

describe("dct del navegador", () => {
  it("da DC = 8c en un bloque constante", () => {
    const block = new Float64Array(64).fill(3)
    const coefficients = dct8(block)
    expect(coefficients[0]).toBeCloseTo(24, 8)
    for (let index = 1; index < coefficients.length; index += 1) expect(coefficients[index]).toBeCloseTo(0, 8)
    const restored = idct8(coefficients)
    for (let index = 0; index < restored.length; index += 1) expect(restored[index]).toBeCloseTo(3, 8)
  })
})
