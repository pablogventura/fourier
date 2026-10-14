import type { Coefficient } from "../math/series.ts"

type Sample = { re: number; im: number }

export function signedFrequency(bin: number, sampleCount: number): number {
  if (sampleCount % 2 === 0 && bin === sampleCount / 2) return sampleCount / 2
  if (bin < sampleCount / 2) return bin
  return bin - sampleCount
}

function swap(values: Float64Array, left: number, right: number) {
  const saved = values[left]
  values[left] = values[right]
  values[right] = saved
}

function fftRadix2(real: Float64Array, imag: Float64Array) {
  const count = real.length
  for (let index = 1, reversed = 0; index < count; index += 1) {
    let bit = count >> 1
    for (; reversed & bit; bit >>= 1) reversed ^= bit
    reversed ^= bit
    if (index < reversed) {
      swap(real, index, reversed)
      swap(imag, index, reversed)
    }
  }
  for (let length = 2; length <= count; length <<= 1) {
    const angle = (-2 * Math.PI) / length
    const stepReal = Math.cos(angle)
    const stepImag = Math.sin(angle)
    for (let start = 0; start < count; start += length) {
      let twiddleReal = 1
      let twiddleImag = 0
      const half = length >> 1
      for (let offset = 0; offset < half; offset += 1) {
        const evenReal = real[start + offset]
        const evenImag = imag[start + offset]
        const oddIndex = start + offset + half
        const oddReal = real[oddIndex] * twiddleReal - imag[oddIndex] * twiddleImag
        const oddImag = real[oddIndex] * twiddleImag + imag[oddIndex] * twiddleReal
        real[start + offset] = evenReal + oddReal
        imag[start + offset] = evenImag + oddImag
        real[oddIndex] = evenReal - oddReal
        imag[oddIndex] = evenImag - oddImag
        const nextReal = twiddleReal * stepReal - twiddleImag * stepImag
        twiddleImag = twiddleReal * stepImag + twiddleImag * stepReal
        twiddleReal = nextReal
      }
    }
  }
}

function ifftRadix2(real: Float64Array, imag: Float64Array) {
  const count = real.length
  for (let index = 0; index < count; index += 1) imag[index] = -imag[index]
  fftRadix2(real, imag)
  for (let index = 0; index < count; index += 1) {
    real[index] /= count
    imag[index] = -imag[index] / count
  }
}

function fftBluestein(real: Float64Array, imag: Float64Array) {
  const count = real.length
  let convolution = 1
  while (convolution < count * 2 - 1) convolution <<= 1
  const chirpReal = new Float64Array(count)
  const chirpImag = new Float64Array(count)
  for (let index = 0; index < count; index += 1) {
    const angle = (Math.PI * index * index) / count
    chirpReal[index] = Math.cos(angle)
    chirpImag[index] = -Math.sin(angle)
  }
  const leftReal = new Float64Array(convolution)
  const leftImag = new Float64Array(convolution)
  const rightReal = new Float64Array(convolution)
  const rightImag = new Float64Array(convolution)
  for (let index = 0; index < count; index += 1) {
    leftReal[index] = real[index] * chirpReal[index] - imag[index] * chirpImag[index]
    leftImag[index] = real[index] * chirpImag[index] + imag[index] * chirpReal[index]
    rightReal[index] = chirpReal[index]
    rightImag[index] = -chirpImag[index]
    if (index > 0) {
      rightReal[convolution - index] = chirpReal[index]
      rightImag[convolution - index] = -chirpImag[index]
    }
  }
  fftRadix2(leftReal, leftImag)
  fftRadix2(rightReal, rightImag)
  for (let index = 0; index < convolution; index += 1) {
    const nextReal = leftReal[index] * rightReal[index] - leftImag[index] * rightImag[index]
    const nextImag = leftReal[index] * rightImag[index] + leftImag[index] * rightReal[index]
    leftReal[index] = nextReal
    leftImag[index] = nextImag
  }
  ifftRadix2(leftReal, leftImag)
  for (let index = 0; index < count; index += 1) {
    const nextReal = leftReal[index] * chirpReal[index] - leftImag[index] * chirpImag[index]
    const nextImag = leftReal[index] * chirpImag[index] + leftImag[index] * chirpReal[index]
    real[index] = nextReal
    imag[index] = nextImag
  }
}

function forwardDft(samples: Sample[]): Sample[] {
  const count = samples.length
  const real = new Float64Array(count)
  const imag = new Float64Array(count)
  for (let index = 0; index < count; index += 1) {
    real[index] = samples[index].re
    imag[index] = samples[index].im
  }
  if (count > 1 && (count & (count - 1)) === 0) fftRadix2(real, imag)
  else if (count > 1) fftBluestein(real, imag)
  const out: Sample[] = []
  for (let index = 0; index < count; index += 1) out.push({ re: real[index] / count, im: imag[index] / count })
  return out
}

function seriesFromSamples(samples: Sample[], center: boolean): { offset: Sample; coefficients: Coefficient[] } {
  let meanReal = 0
  let meanImag = 0
  if (center) {
    for (const sample of samples) {
      meanReal += sample.re
      meanImag += sample.im
    }
    meanReal /= samples.length
    meanImag /= samples.length
  }
  const centered = samples.map((sample) => ({ re: sample.re - meanReal, im: sample.im - meanImag }))
  const bins = forwardDft(centered)
  const coefficients = bins.map((bin, index) => {
    const frequency = signedFrequency(index, bins.length)
    return {
      frequency,
      real: bin.re,
      imag: bin.im,
      amplitude: Math.hypot(bin.re, bin.im),
      phase: Math.atan2(bin.im, bin.re),
    }
  })
  coefficients.sort((left, right) => left.frequency - right.frequency)
  return { offset: { re: meanReal, im: meanImag }, coefficients }
}

export type SourceSpectrum = {
  sampleCount: number
  centered: boolean
  offsetX: number
  offsetY: number
  coefficients: Coefficient[]
  xCoefficients: Coefficient[]
  yCoefficients: Coefficient[]
}

export function analyzeSourcePoints(points: Array<[number, number]>, center: boolean): SourceSpectrum {
  if (points.length < 4) throw new Error("Hacen falta al menos 4 puntos (x, y).")
  const complex = points.map(([x, y]) => ({ re: x, im: -y }))
  const xSamples = points.map(([x]) => ({ re: x, im: 0 }))
  const ySamples = points.map(([, y]) => ({ re: y, im: 0 }))
  const zSeries = seriesFromSamples(complex, center)
  const xSeries = seriesFromSamples(xSamples, center)
  const ySeries = seriesFromSamples(ySamples, center)
  let offsetX = 0
  let offsetY = 0
  if (center) {
    for (const [x, y] of points) {
      offsetX += x
      offsetY += y
    }
    offsetX /= points.length
    offsetY /= points.length
  }
  return {
    sampleCount: points.length,
    centered: center,
    offsetX,
    offsetY,
    coefficients: zSeries.coefficients,
    xCoefficients: xSeries.coefficients,
    yCoefficients: ySeries.coefficients,
  }
}
