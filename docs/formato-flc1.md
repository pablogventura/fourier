# Formato FLC1

FLC1 guarda un contorno como coeficientes de Fourier. No guarda pixeles ni la polilínea original. Little-endian.

La reconstrucción, con t = n / N y n = 0 .. N-1, es

```text
z(t) = suma_k  c_k * exp(2 π i k t)
x = offset_x + scale_x * Re(z)
y = offset_y - scale_y * Im(z)
```

`y` crece hacia abajo, como en una imagen. El imaginario de `z` apunta hacia arriba, así una frecuencia positiva gira en sentido antihorario en la pantalla. Si el archivo está centrado, el offset es el centroide y la traslación no está en los coeficientes. Si no, el offset es cero y esa traslación vive en la frecuencia 0.

## Cabecera, 56 bytes

```text
0   4   marca ASCII "FLC1"
4   2   versión uint16, hoy 1
6   2   flags uint16
8   4   N, cantidad de muestras uint32
12  4   cantidad de coeficientes uint32
16  8   offset_x float64
24  8   offset_y float64
32  8   scale_x float64
40  8   scale_y float64
48  8   paso de cuantización float64, o 0 si no hay cuantización
```

Flags:

- bit 0: el contorno es cerrado
- bit 1: los coeficientes están cuantizados a int16
- bit 2: sin cuantizar, cada parte es float64. Si este bit está en 0 y no hay cuantización, cada parte es float32
- bit 3: el contorno se centró antes de la FFT

## Registros

Uno por coeficiente, en el orden en que se guardaron.

- cuantizado: int32 frecuencia, int16 real, int16 imag. El valor es el entero por el paso.
- float64: int32 frecuencia, float64 real, float64 imag
- float32: int32 frecuencia, float32 real, float32 imag

La frecuencia es la firmada, en ciclos por período. Con N par, el bin de Nyquist se escribe como +N/2 y no se duplica.

## Qué mide el tamaño

`N * 16` bytes es el arreglo crudo de puntos en float64, sin cabecera. Otro tanto son todos los coeficientes complejos, también en crudo. `encoded_bytes` es el archivo FLC1, cabecera incluida. Esas cuentas no son intercambiables, y ninguna es el tamaño de un JPEG de la imagen de la que salió el contorno.
