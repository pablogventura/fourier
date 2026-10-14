# FourierLab

Laboratorio para ver una serie de Fourier dibujar un contorno, medir qué se pierde al quedarse con pocos coeficientes, y comparar eso con una DCT por bloques al estilo JPEG.

El sitio publicado corre entero en el navegador: [https://pablogventura.github.io/fourier/](https://pablogventura.github.io/fourier/). No hace falta el backend para usarlo.

El backend en Python 3.12 queda como implementación de referencia, con las mismas convenciones y con sus pruebas. El video del sitio es WebM. El backend, si se llama por su API, puede escribir MP4.

## Instalación en Ubuntu

Hace falta Node 22 o posterior para la interfaz. Python 3.12 hace falta solo para las pruebas del backend. ffmpeg hace falta solo si vas a pedir un MP4 a esa API.

```bash
cd /ruta/a/fourier
python3.12 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn numpy scipy opencv-python-headless pillow python-multipart pytest httpx
cd frontend
npm install
```

## Ejecución

La interfaz no llama al backend:

```bash
cd frontend
npm install
npm run dev
```

Abrí http://127.0.0.1:5173.

El servicio de referencia, si lo querés correr aparte:

```bash
source .venv/bin/activate
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Pruebas

```bash
source .venv/bin/activate
cd backend && python -m pytest
cd ../frontend && npm test
```

## Convención de la serie

Las muestras están en coordenadas de imagen: x hacia la derecha, y hacia abajo. La señal compleja es

```text
z[n] = x[n] - i y[n]
c_k = FFT(z)[k] / N
z(t) = suma_k c_k exp(2 π i k t)
```

`FFT` es `numpy.fft.fft`. La frecuencia es la firmada: el bin `k` vale `k` si `k < N/2` y `k - N` si `k > N/2`. Con N par, el bin de Nyquist queda como `+N/2`. En las muestras enteras `t = n/N`, eso coincide con `-N/2`. Entre muestras no, y la animación usa la frecuencia firmada para que un círculo horario gire en sentido horario en el plano complejo.

En la pantalla, y hacia abajo y el imaginario hacia arriba hacen que una frecuencia positiva se vea antihoraria.

Centrar, activo por defecto, resta el centroide antes de la FFT. El offset se vuelve a sumar al dibujar y se guarda en el archivo. Apagarlo deja la traslación en el primer círculo, el de frecuencia 0.

Dos recortes, con tope de 500 coeficientes:

- Simétrico: frecuencias de `-K` a `K`. El control pide una cantidad de componentes y se ajusta al impar `2K+1`.
- Magnitud: los K coeficientes de mayor módulo.

El orden de la suma no importa. El orden del dibujo sí: primero k = 0, después |k| creciente, y a igual módulo la frecuencia positiva antes que la negativa.

La FFT se calcula una vez al elegir el contorno. Mover K, o avanzar un fotograma, solo filtra o evalúa esos coeficientes.

## Contornos

Cada curva es una señal. No se concatenan contornos desconectados ni subtrazados de un SVG. Un agujero y el borde exterior son dos series. Un trazo abierto se marca como tal: la DFT igual lo trata como un período, la reconstrucción cierra el salto y ahí aparece Gibbs.

El remuestreo reparte N puntos con la misma longitud de arco. El parámetro de la serie es ese índice, no el tiempo que tardaste en dibujar.

En SVG se leen path, rect, circle, ellipse, polygon, polyline y line, con las transformaciones del elemento y de sus grupos. En el backend, el comando de arco A se parte en cúbicas de a lo sumo 90 grados. En el sitio, ese arco lo muestrea el navegador.

En una imagen raster, si hay alfa útil la máscara sale de la opacidad. Si no, de la luminancia. La polaridad automática invierte el umbral cuando el borde de la imagen quedó del lado de la figura, que es el caso de un dibujo oscuro sobre fondo claro.

## Compresión del contorno

El archivo [FLC1](docs/formato-flc1.md) guarda coeficientes, no pixeles. Las métricas separan el arreglo crudo en float64 del archivo codificado. RMSE y error máximo comparan puntos con el mismo parámetro t. Hausdorff es geométrica, sin esa correspondencia.

Eso no comprime la imagen de la que salió el contorno.

## Laboratorio DCT

La segunda pestaña aplica la DCT-II ortonormal en bloques de 8x8. Para N = 8 esa escala coincide con la DCT del JPEG, así que la tabla de luminancia y la calidad al estilo IJG se comportan como uno espera. Antes de transformar se resta 128. En el backend la cuenta la hace SciPy. En el sitio, la misma escala se calcula en el navegador.

No es un codificador JPEG: no hay paso a YCbCr, no hay submuestreo 4:2:0 y no hay codificación entrópica. La diferencia en pantalla está amplificada por 4.

## Exportación

SVG y PNG del contorno reconstruido, GIF y WebM de los epiciclos, CSV y JSON de coeficientes, FLC1, y CSV de métricas junto con la curva de tasa-distorsión. GIF y WebM usan la misma suma de fasores que el canvas. El cuadro del sitio es de 640 por 480 px y 72 fotogramas. En el navegador, Hausdorff compara como máximo 256 puntos de cada curva.
