// =====================================================================
//  app.js  –  Varios rostros + varias manos + contador de dedos + dibujo
//  Librería: MediaPipe Tasks Vision (Google), cargada desde un CDN.
// =====================================================================

// ═══════════════════════════════════════════════════════════════════════
// GUÍA RÁPIDA PARA EXPONER ESTE ARCHIVO
// ═══════════════════════════════════════════════════════════════════════
// ¿QUÉ HACE?
//   Usa la cámara para detectar rostros y manos, cuenta los dedos de cada mano
//   y reacciona según el MODO elegido (gestos, dibujo, quiz, vigilancia, etc.).
//
// IDEA CENTRAL
//   Una inteligencia artificial ya entrenada (MediaPipe, de Google) mira cada
//   imagen del video y nos devuelve DATOS: dónde hay rostros y 21 puntos por mano.
//   Nuestro código no "ve": solo convierte esos datos en acciones.
//
// FLUJO DEL PROGRAMA
//   1. El usuario hace clic en "Activar cámara"          (sección 13)
//   2. Se cargan los modelos de IA                       (sección 5)
//   3. Se enciende la cámara                             (sección 6)
//   4. Arranca el BUCLE, que se repite decenas de veces por segundo (sección 12):
//        a) pide a la IA los rostros y las manos del cuadro actual
//        b) dibuja recuadros y esqueletos encima del video
//        c) cuenta los dedos de cada mano                (sección 7)
//        d) entrega los datos a la función del modo activo (secciones 9 a 11)
//
// CAPAS EN PANTALLA (de abajo hacia arriba)
//   <video> con la cámara  <  <canvas id="dibujo"> con los trazos  <  <canvas id="lienzo"> con rostros y manos
//
// MAPA DEL ARCHIVO
//   1-2 librería y constantes | 3-4 referencias al HTML y estado | 5-6 modelos y cámara
//   7 contar dedos | 8 texto en el canvas | 9 modo gestos | 10 fotos | 11 los demás modos
//   12 bucle principal | 13 arranque | 14 pestañas
//
// VOCABULARIO
//   frame = una imagen del video | landmark = un punto de la mano (hay 21 por mano)
//   canvas = superficie de la página donde se dibuja con código | DOM = la página como objetos
//
// PREGUNTAS PROBABLES
//   ¿Por qué no Dlib/Python? No funcionó en el equipo; MediaPipe corre en el navegador sin instalar nada.
//   ¿Cómo sabe qué dedos están arriba? Compara distancias entre puntos de la mano (sección 7).
//   ¿Por qué esperar 6 frames para confirmar un gesto? Para ignorar errores de un solo cuadro.
//   ¿Necesita internet? Sí: la librería y los modelos se descargan al abrir la página.
//   ¿Se guardan datos de las personas? No se envía nada a ningún servidor; las fotos quedan en el navegador.
// 1) IMPORTAR LA LIBRERÍA ---------------------------------------------
// "import" trae código desde otra dirección. Por eso index.html carga este archivo con
// type="module". Pedimos solo las piezas que usamos para no cargar de más.
import {
  FilesetResolver, // motor WebAssembly que corre los modelos
  FaceDetector,    // detector de rostros (detecta TODOS los que haya)
  FaceLandmarker,  // puntos del rostro + expresiones (sirve para detectar la sonrisa)
  HandLandmarker,  // detector de manos (21 puntos por mano)
  DrawingUtils     // ayudante para dibujar puntos y líneas
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// 2) CONSTANTES --------------------------------------------------------
// Aquí van los valores que podrías querer ajustar. Ponerlos arriba, con nombre, evita
// "números mágicos" escondidos en medio del código: para cambiar algo se edita UNA línea.
// Las direcciones (WASM, modelos) apuntan a los archivos de la IA que se descargan al iniciar.
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODELO_ROSTRO =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const MODELO_MANO =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const MODELO_LANDMARKS =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const ESPERA_VACIO_MS = 3000; // vigilancia: segundos sin nadie antes de que una aparición cuente como alerta
const TAM_BORRADOR = 44;      // diámetro del borrador (dos dedos)
let vision;                   // motor de MediaPipe, compartido por todos los modelos
const MAX_MANOS = 4;         // 4 manos = 2 personas x 2 manos. Si la PC va lenta, bájalo a 2.
const PADDING = 60;          // margen alrededor de cada rostro
const FRAMES_ESTABLES = 6;   // frames seguidos para confirmar un gesto
const ESPERA_FOTO_MS = 2000; // mínimo 2 s entre fotos

// Modo dibujo
const COLORES_PINCEL = ["#ffffff", "#ff4d4d", "#4da3ff", "#43d17a", "#f5d547", "#c77dff"]; // 1 por mano
const GROSOR = 6;
const SUAVIZADO = 0.45;      // 0 a 1: más bajo = trazo más suave pero con retraso
const RADIO_EMPAREJAR = 120; // px: máximo que se mueve un dedo entre frames para ser "el mismo lápiz"
const FRAMES_BORRAR = 25;    // frames con 5 dedos para borrar todo (~0.8 s)

// Acción de cada número (el color cambia el borde del video y del panel)
// Objeto usado como "tabla de búsqueda": ACCIONES[3] devuelve la acción del número 3.
// Es más corto y ordenado que escribir seis bloques if/else.
const ACCIONES = {
  0: { color: "#8a97a3", texto: "Puño: modo pausa." },
  1: { color: "#4da3ff", texto: "1 dedo: borde azul." },
  2: { color: "#43d17a", texto: "2 dedos: borde verde." },
  3: { color: "#ff9f43", texto: "3 dedos: borde naranja." },
  4: { color: "#ff6fae", texto: "4 dedos: borde rosa." },
  5: { color: "#f5d547", texto: "5 dedos: foto de todos los rostros." }
};
const TEXTO_INICIAL =
  "Muestra una o más manos. Con varias manos, la página reacciona cuando todas muestran el mismo número.";

// 3) REFERENCIAS AL HTML ----------------------------------------------
// Guardamos en variables los elementos del HTML con los que vamos a trabajar. Se buscan
// UNA vez aquí (buscarlos es lento) y luego se reutilizan. "$" es solo un atajo para
// document.getElementById.
const video = document.getElementById("video");
const lienzo = document.getElementById("lienzo");       // rostros y manos (se borra cada frame)
const ctx = lienzo.getContext("2d");
const dibujo = document.getElementById("dibujo");       // solo trazos (NO se borra solo)
const ctxDibujo = dibujo.getContext("2d");
const dibujante = new DrawingUtils(ctx);
const btnIniciar = document.getElementById("btnIniciar");
const $ = (id) => document.getElementById(id);   // atajo para no repetir document.getElementById
const selModo = $("selModo"), btnBorrar = $("btnBorrar"), btnDeshacer = $("btnDeshacer");
const inpColor = $("inpColor"), btnColorAuto = $("btnColorAuto"), inpGrosor = $("inpGrosor");
const btnPng = $("btnPng"), btnPngCam = $("btnPngCam"), btnZip = $("btnZip"), btnCsv = $("btnCsv");
const btnClasico = $("btnClasico"), txtClasico = $("txtClasico");
const txtPregunta = $("txtPregunta"), txtPuntos = $("txtPuntos"), btnPregunta = $("btnPregunta");
const btnVig = $("btnVig"), txtVig = $("txtVig"), txtSonrisa = $("txtSonrisa");
const btnPpt = $("btnPpt"), txtPpt = $("txtPpt"), txtPptPuntos = $("txtPptPuntos");
const btnAnt = $("btnAnt"), btnSig = $("btnSig"), selCamara = $("selCamara");
const btnPantalla = $("btnPantalla"), chkEsqueleto = $("chkEsqueleto"), txtFps = $("txtFps"), listaRegistro = $("registro");
const estado = document.getElementById("estado");
const numDedos = document.getElementById("numDedos");
const detalleManos = document.getElementById("detalleManos");
const accionTexto = document.getElementById("accionTexto");
const galeria = document.getElementById("galeria");

// 4) VARIABLES DE ESTADO ----------------------------------------------
// El "estado" es la memoria del programa: datos que cambian mientras se usa (modo actual,
// trazos dibujados, fotos tomadas...). Se declaran con let porque su valor se reasigna.
// Las que no cambian de valor se declaran con const.
let detectorRostro, detectorMano;
let ultimoTiempoVideo = -1;
let cajasRostro = [];        // un recuadro (con padding) por cada rostro del frame actual
let conteoCandidato = -1, repeticiones = 0, conteoConfirmado = -1;
let ultimaFoto = 0;
let modoDibujo = false;
let lapices = [];            // lápices activos: [{x, y, color}] (uno por mano que dibuja)
let framesPalma = 0;         // frames seguidos con alguna mano abierta (para borrar)

// Estado de las funciones nuevas
const MAX_CLASICO = 10;      // el programa de Python se detenía a las 10 imágenes
let modo = "gestos";         // gestos | dibujo | clasico | quiz
let trazos = [];             // cada trazo guardado como lista de puntos (permite DESHACER)
let grosor = 6, colorManual = null;   // null = un color distinto por persona
let fotos = [], registros = [];       // fotos tomadas (para el ZIP) y bitácora (para el CSV)
const clasico = { activo: false, n: 0, ultimo: 0 };
const quiz = { resp: -1, frames: 0, bloqueo: 0, puntos: {} };
const fps = { n: 0, t: 0 };
let rostrosEstables = 0, rostrosRegistrados = -1;

// 5) CARGAR LOS MODELOS -----------------------------------------------
// Carga los dos modelos de IA. Es "async" porque descargar toma tiempo; "await" hace que el
// código espere a que termine antes de seguir.
//   - FilesetResolver prepara el motor (WebAssembly) que ejecuta los modelos.
//   - runningMode "VIDEO" = analizar imágenes seguidas (no una foto suelta).
//   - delegate "GPU" usa la tarjeta gráfica (más rápido); si falla, el try/catch reintenta con "CPU".
//   - numHands limita cuántas manos detecta: más manos = más trabajo para el equipo.
async function crearDetectores() {
  estado.textContent = "Cargando modelos de IA...";
  vision = await FilesetResolver.forVisionTasks(WASM);
  const crear = async (delegate) => {
    detectorRostro = await FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODELO_ROSTRO, delegate },
      runningMode: "VIDEO"
    });
    detectorMano = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODELO_MANO, delegate },
      runningMode: "VIDEO",
      numHands: MAX_MANOS     // máximo de manos a la vez
    });
  };
  try { await crear("GPU"); } catch { await crear("CPU"); }
}

// 6) ENCENDER LA CÁMARA -----------------------------------------------
// Pide permiso y enciende la cámara.
//   - getUserMedia muestra el aviso de permiso del navegador y devuelve el flujo de video.
//   - "ideal" pide un tamaño pero acepta otro si la cámara no lo tiene.
//   - Los canvas se igualan al tamaño REAL del video: si no, los dibujos saldrían desplazados.
//   - La cámara frontal se muestra en espejo (como un selfie); la trasera no.
async function encenderCamara(idCamara) {
  const anterior = video.srcObject;   // guardamos la cámara anterior: solo se apaga si la nueva abre bien
  // facingMode "user" = cámara frontal en el celular. "ideal" = pide ese tamaño, pero acepta otro.
  const flujo = await navigator.mediaDevices.getUserMedia({
    video: { ...(idCamara ? { deviceId: { exact: idCamara } } : { facingMode: "user" }), width: { ideal: 640 }, height: { ideal: 480 } }
  });
  anterior?.getTracks().forEach(t => t.stop());   // la nueva funcionó: ahora sí apagamos la anterior
  video.srcObject = flujo;
  await video.play();
  // Algunas cámaras USB tardan en entregar su primer cuadro. Si seguimos sin esperar, videoWidth
  // vale 0, los canvas quedan de tamaño 0 y NO se detecta ni se dibuja nada (parece que "no reconoce").
  for (let i = 0; i < 50 && !video.videoWidth; i++) await new Promise(r => setTimeout(r, 100));
  if (!video.videoWidth) throw new Error("La cámara no entregó imagen");
  // Los dos canvas deben tener la MISMA resolución que el video, o todo saldría desplazado.
  lienzo.width = dibujo.width = video.videoWidth;
  lienzo.height = dibujo.height = video.videoHeight;
  // Adaptamos la caja a la proporción REAL de la cámara (en un celular vertical es 3:4, no 4:3).
  // Si no, el video se estiraría y los dibujos quedarian desalineados.
  const escenario = document.querySelector(".escenario");
  escenario.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  escenario.style.setProperty("--ratio", video.videoWidth / video.videoHeight);   // lo usa el CSS en pantalla completa

  // Detectar si la cámara activa es trasera para no voltearla en espejo
  const track = flujo.getVideoTracks()[0];
  const settings = track?.getSettings ? track.getSettings() : {};
  console.log("Cámara activa:", track?.label, settings);   // aparece en la consola (F12) para diagnosticar
  const label = (track?.label || "").toLowerCase();
  const esTrasera = settings.facingMode === "environment" || label.includes("back") || label.includes("trasera") || label.includes("rear");
  escenario.classList.toggle("escenario--trasera", esTrasera);

  trazos = []; lapices = [];   // al cambiar de cámara el canvas se reinicia
  await llenarCamaras();
}

// 7) DEDOS ESTIRADOS (por mano) ----------------------------------------
// Puntos usados: 0 muñeca | 4,8,12,16,20 puntas | 3 articulación del pulgar
// 6,10,14,18 articulación media de índice, medio, anular, meñique | 17 base del meñique.
// Un dedo está estirado si su punta está más lejos de la muñeca que su articulación.
// EL CORAZÓN DEL CONTADOR DE DEDOS.
// Regla: un dedo está ESTIRADO si su punta está más LEJOS de la muñeca que su articulación
// media. Con el dedo doblado, la punta regresa hacia la palma y queda más cerca.
//   - Se usan distancias (no "arriba/abajo") para que funcione con la mano inclinada.
//   - Los puntos vienen entre 0 y 1; se pasan a píxeles para que las distancias no se deformen.
//   - El PULGAR es especial: se abre hacia el costado, así que se compara contra la base del meñique.
// Devuelve [pulgar, índice, medio, anular, meñique] con true/false. Contar los true da el número.
function dedosEstirados(puntos) {
  // Pasamos de 0-1 a píxeles para no deformar las distancias.
  const p = puntos.map(pt => ({ x: pt.x * lienzo.width, y: pt.y * lienzo.height }));
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  // Pulgar: se compara contra la base del meñique porque se abre hacia el lado.
  const pulgar = dist(p[4], p[17]) > dist(p[3], p[17]);
  const pares = [[8, 6], [12, 10], [16, 14], [20, 18]];
  const otros = pares.map(([punta, art]) => dist(p[punta], p[0]) > dist(p[art], p[0]));
  return [pulgar, ...otros];   // [pulgar, índice, medio, anular, meñique]
}

// 8) TEXTO SOBRE EL CANVAS ------------------------------------------------
// Si es cámara frontal está en espejo por CSS, así que volteamos el texto con scale(-1, 1).
// Si es cámara trasera, escribimos en orientación normal.
// Escribe texto sobre el canvas (ej. "PERSONA 1"). El video está volteado en espejo con CSS, y el
// texto se voltearía también y se leería al revés; scale(-1, 1) lo vuelve a voltear.
// save()/restore() guardan y devuelven la configuración del pincel para no afectar otros dibujos.
function dibujarEtiqueta(texto, x, y, fondo) {
  texto = texto.toUpperCase();   // el canvas NO hereda el CSS: hay que pasar a mayúsculas aquí
  const esTrasera = document.querySelector(".escenario")?.classList.contains("escenario--trasera");
  ctx.save();
  ctx.translate(x, y);
  if (!esTrasera) {
    ctx.scale(-1, 1);
  }
  ctx.font = "20px Oxygene, 'Segoe UI', sans-serif";   // misma fuente que la página
  const ancho = ctx.measureText(texto).width + 14;
  ctx.fillStyle = fondo;
  ctx.fillRect(-ancho / 2, -26, ancho, 26);
  ctx.fillStyle = "#fff";
  ctx.textAlign = "center";
  ctx.fillText(texto, 0, -6);
  ctx.restore();
}

// 9) ACCIONES POR GESTO (modo normal) ------------------------------------
// Ejecuta lo que corresponde al número confirmado: cambia el color de acento (una variable CSS,
// así todo el borde cambia con una sola línea) y muestra el texto. Con 5 dedos toma foto.
function aplicarAccion(n) {
  const accion = ACCIONES[n];
  document.documentElement.style.setProperty("--acento", accion.color);
  accionTexto.textContent = accion.texto;
  if (n === 5) tomarFotos();
}

// Con varias manos usamos "consenso": la acción solo se activa si TODAS las manos
// muestran el mismo número. Así dos personas con gestos distintos no se pisan.
// MODO GESTOS. Dos ideas importantes:
//   1) CONSENSO: con varias manos, la acción solo se activa si TODAS muestran el mismo número,
//      para que dos personas con gestos distintos no se pisen.
//   2) ANTI-PARPADEO: el número debe repetirse FRAMES_ESTABLES cuadros seguidos. Si una sola imagen
//      sale mal, no pasa nada, porque el contador de repeticiones se reinicia.
// conteoConfirmado recuerda el último gesto aplicado para no repetir la acción en cada cuadro.
function procesarGestos(manos) {
  if (manos.length === 0) { conteoCandidato = -1; repeticiones = 0; conteoConfirmado = -1; return; }

  const primero = manos[0].n;
  const consenso = manos.every(m => m.n === primero) ? primero : -1;

  // Anti-parpadeo: el valor debe repetirse varios frames seguidos.
  repeticiones = (consenso === conteoCandidato) ? repeticiones + 1 : 1;
  conteoCandidato = consenso;

  if (consenso === -1) { conteoConfirmado = -1; return; }   // no hay acuerdo
  if (repeticiones >= FRAMES_ESTABLES && consenso !== conteoConfirmado) {
    conteoConfirmado = consenso;
    aplicarAccion(consenso);
  }
  if (consenso === 5 && repeticiones >= FRAMES_ESTABLES) tomarFotos(); // repite cada 2 s si sigues
}

// 10) FOTOS: una por cada rostro detectado ---------------------------------
// Recorta UN rostro, lo guarda en la lista (para el ZIP) y lo muestra en la galería
// Recorta UN rostro del video: se dibuja en un canvas temporal solo la zona del recuadro
// (drawImage con 9 parámetros = origen, tamaño de corte y destino). Luego se convierte a imagen
// (JPG en texto base64), se guarda en la lista "fotos" (para el ZIP) y se muestra en la galería.
function guardarFoto(caja, i) {
  const { x, y, ancho, alto } = caja;
  const recorte = document.createElement("canvas");
  recorte.width = ancho;
  recorte.height = alto;
  recorte.getContext("2d").drawImage(video, x, y, ancho, alto, 0, 0, ancho, alto);
  const url = recorte.toDataURL("image/jpeg", 0.9);
  const nombre = `persona${i + 1}_${Date.now()}.jpg`;
  fotos.push({ nombre, url });

  const enlace = document.createElement("a");
  enlace.href = url;
  enlace.download = nombre;
  const img = document.createElement("img");
  img.src = url;
  img.alt = `Foto de la persona ${i + 1}`;
  enlace.appendChild(img);
  galeria.querySelector(".vacio")?.remove();
  galeria.prepend(enlace);
  registrar(`Foto guardada: persona ${i + 1}`);
}

// Toma una foto de TODOS los rostros visibles, con espera de 2 s entre tomas (ESPERA_FOTO_MS)
// para no llenar la galería con decenas de fotos iguales.
function tomarFotos() {
  const ahora = performance.now();
  if (cajasRostro.length === 0 || ahora - ultimaFoto < ESPERA_FOTO_MS) return;
  ultimaFoto = ahora;
  cajasRostro.forEach(guardarFoto);
}

// 11) MODO DIBUJO (varias personas) -------------------------------------------
// Problema nuevo: MediaPipe NO dice "esta es la mano 1 del frame anterior".
// Solución: emparejamos cada punta de dedo con el lápiz más cercano del frame anterior.
// Si está a menos de RADIO_EMPAREJAR px, es el mismo lápiz y seguimos su trazo.
// MODO DIBUJO CON VARIAS PERSONAS. Pasos:
//   1) Se clasifica cada mano por su gesto: solo índice = dibujar | índice + medio = borrar |
//      "L" con el pulgar = cambiar grosor. El tamaño de la palma sirve de regla para comparar.
//   2) PROBLEMA: la IA no dice "esta es la misma mano del cuadro anterior". SOLUCIÓN: se empareja
//      cada punta de dedo con el lápiz más cercano del cuadro anterior (a menos de RADIO_EMPAREJAR px).
//   3) Si hay pareja, se traza una LÍNEA desde el punto anterior al actual (no puntos sueltos, o
//      quedarían huecos al mover rápido). Si no hay, se crea un lápiz nuevo con un color libre.
//   4) SUAVIZADO: el punto avanza solo una parte hacia la punta real, para quitar el temblor.
//   5) Cada trazo se guarda como lista de puntos: eso permite DESHACER y volver a dibujar todo.
//   6) Borrar usa "destination-out": en vez de pintar, quita lo pintado.
function procesarDibujo(manos) {
  // Borrar: cualquier mano abierta mantenida (se dispara una sola vez por mantenida).
  framesPalma = manos.some(m => m.n === 5) ? framesPalma + 1 : 0;
  if (framesPalma === FRAMES_BORRAR) borrarDibujo();

  // Solo cuentan las manos con SOLO el índice arriba (el pulgar se ignora: es inestable).
  const activas = manos.map(m => {
    const [, indice, medio, anular, menique] = m.estirados;
    const P = (i) => ({ x: m.puntos[i].x * dibujo.width, y: m.puntos[i].y * dibujo.height });
    const d = (p, q) => Math.hypot(p.x - q.x, p.y - q.y);
    const palma = d(P(0), P(9));                         // tamaño de la mano: sirve de "regla"
    const pulgarAbierto = d(P(4), P(5)) > palma * 0.8;   // pulgar bien separado = forma de "L"
    const soloIndice = indice && !medio && !anular && !menique;
    // Gesto "L" (pulgar + índice): qué tan separadas están las puntas define el grosor
    if (soloIndice && pulgarAbierto) {
      grosor = Math.round(Math.min(24, Math.max(2, 2 + (d(P(4), P(8)) / palma - 0.2) * 30)));
      inpGrosor.value = grosor;
    }
    return { punta: P(8), dibuja: soloIndice && !pulgarAbierto,
             borra: indice && medio && !anular && !menique,   // índice + medio = borrador
             lapiz: null };
  });

  // Pasada 1: emparejar con los lápices del frame anterior.
  const usados = new Set();
  for (const a of activas.filter(a => a.dibuja || a.borra)) {
    let mejor = -1, mejorDist = RADIO_EMPAREJAR;
    lapices.forEach((l, i) => {
      if (usados.has(i) || l.trazo.borrar !== !!a.borra) return;   // el borrador solo sigue a borradores
      const d = Math.hypot(l.x - a.punta.x, l.y - a.punta.y);
      if (d < mejorDist) { mejorDist = d; mejor = i; }
    });
    if (mejor >= 0) {
      usados.add(mejor);
      const prev = lapices[mejor];
      // Suavizado: avanzamos solo una parte hacia la punta real para quitar el temblor.
      const nuevo = { x: prev.x + (a.punta.x - prev.x) * SUAVIZADO,
                      y: prev.y + (a.punta.y - prev.y) * SUAVIZADO,
                      color: prev.color, trazo: prev.trazo };
      prev.trazo.puntos.push({ x: nuevo.x, y: nuevo.y });   // guardamos el punto para poder deshacer
      // Línea desde el punto anterior al actual (si no, quedarían huecos al mover rápido).
      ctxDibujo.beginPath();
      ctxDibujo.moveTo(prev.x, prev.y);
      ctxDibujo.lineTo(nuevo.x, nuevo.y);
      ctxDibujo.strokeStyle = nuevo.color;
      ctxDibujo.lineWidth = prev.trazo.grosor;
      ctxDibujo.lineCap = ctxDibujo.lineJoin = "round";
      // "destination-out" = en vez de pintar, QUITA lo pintado (así funciona un borrador)
      ctxDibujo.globalCompositeOperation = prev.trazo.borrar ? "destination-out" : "source-over";
      ctxDibujo.stroke();
      ctxDibujo.globalCompositeOperation = "source-over";
      a.lapiz = nuevo;
    }
  }

  // Pasada 2: las manos sin lápiz previo estrenan uno, con un color que nadie esté usando.
  for (const a of activas.filter(a => (a.dibuja || a.borra) && !a.lapiz)) {
    const enUso = activas.filter(o => o.lapiz).map(o => o.lapiz.color);
    const color = colorManual ?? COLORES_PINCEL.find(c => !enUso.includes(c)) ?? COLORES_PINCEL[0];
    const trazo = { color, grosor: a.borra ? TAM_BORRADOR : grosor, borrar: !!a.borra, puntos: [{ x: a.punta.x, y: a.punta.y }] };
    trazos.push(trazo);
    a.lapiz = { x: a.punta.x, y: a.punta.y, color, trazo };
  }

  // Los lápices de este frame pasan a ser "los anteriores" del siguiente.
  // Si una mano levanta el lápiz o desaparece, su lápiz se pierde y el trazo se corta.
  lapices = activas.filter(a => a.lapiz).map(a => a.lapiz);

  // Cursor sobre cada índice: LLENO = dibujando, HUECO = lápiz levantado.
  for (const a of activas) {
    ctx.beginPath();
    ctx.arc(a.punta.x, a.punta.y, a.borra ? TAM_BORRADOR / 2 : grosor / 2 + 4, 0, Math.PI * 2);
    if (a.lapiz && !a.borra) { ctx.fillStyle = a.lapiz.color; ctx.fill(); }
    else { ctx.strokeStyle = "#fff"; ctx.lineWidth = 3; ctx.stroke(); }
  }
}

// Limpia el canvas de trazos y vacía la lista de trazos y de lápices.
function borrarDibujo() {
  ctxDibujo.clearRect(0, 0, dibujo.width, dibujo.height);
  lapices = [];
  trazos = [];
}


// 11b) REGISTRO (bitácora) y descargas ---------------------------------------
// Bitácora: guarda cada evento con fecha y hora (para el CSV) y lo muestra en la lista de la
// página. Solo se muestran los últimos 40 para que la lista no crezca sin límite.
function registrar(evento) {
  registros.push({ fecha: new Date().toLocaleString("es-SV"), evento });
  const li = document.createElement("li");
  li.textContent = `${new Date().toLocaleTimeString("es-SV")} - ${evento}`;
  listaRegistro.prepend(li);
  if (listaRegistro.children.length > 40) listaRegistro.lastChild.remove();   // solo se muestran los últimos 40
}

// Descarga cualquier archivo: crea un enlace invisible con el atributo "download" y simula un clic.
// Se reutiliza para el CSV, el ZIP y el PNG del dibujo.
function bajar(url, nombre) {            // descarga cualquier URL como archivo
  const enlace = document.createElement("a");
  enlace.href = url;
  enlace.download = nombre;
  enlace.click();
}

// Exportar el registro a CSV (texto separado por comas, abre en Excel). Las comillas dobles
// dentro del texto se duplican, que es la regla del formato CSV. "\ufeff" al inicio hace que
// Excel lea bien las tildes.
btnCsv.addEventListener("click", () => {
  const filas = registros.map(r => `"${r.fecha}","${r.evento.replace(/"/g, '""')}"`);
  // \ufeff = marca para que Excel lea bien las tildes
  const csv = "\ufeffFecha,Evento\n" + filas.join("\n");
  bajar(URL.createObjectURL(new Blob([csv], { type: "text/csv" })), "registro.csv");
});

// Descargar todas las fotos en un solo ZIP. JSZip arma el archivo en la memoria del navegador;
// "await" espera a que termine de comprimir antes de descargarlo.
btnZip.addEventListener("click", async () => {
  if (fotos.length === 0) return;
  const zip = new JSZip();   // JSZip se carga con un <script> en index.html
  fotos.forEach(f => zip.file(f.nombre, f.url.split(",")[1], { base64: true }));
  bajar(URL.createObjectURL(await zip.generateAsync({ type: "blob" })), "fotos.zip");
});

// 11c) DIBUJO: deshacer, color, grosor y descarga --------------------------------
// Vuelve a pintar todos los trazos guardados desde cero. Se usa al DESHACER: se quita el último
// trazo de la lista y se redibuja lo que queda.
function redibujar() {                   // borra y vuelve a pintar todos los trazos guardados
  ctxDibujo.clearRect(0, 0, dibujo.width, dibujo.height);
  for (const t of trazos) {
    if (t.puntos.length < 2) continue;
    ctxDibujo.beginPath();
    ctxDibujo.moveTo(t.puntos[0].x, t.puntos[0].y);
    t.puntos.slice(1).forEach(p => ctxDibujo.lineTo(p.x, p.y));
    ctxDibujo.strokeStyle = t.color;
    ctxDibujo.lineWidth = t.grosor;
    ctxDibujo.globalCompositeOperation = t.borrar ? "destination-out" : "source-over";
    ctxDibujo.lineCap = ctxDibujo.lineJoin = "round";
    ctxDibujo.stroke();
  }
  ctxDibujo.globalCompositeOperation = "source-over";
}
btnDeshacer.addEventListener("click", () => { trazos.pop(); lapices = []; redibujar(); });
btnBorrar.addEventListener("click", borrarDibujo);
inpColor.addEventListener("input", () => { colorManual = inpColor.value; });
btnColorAuto.addEventListener("click", () => { colorManual = null; });
inpGrosor.addEventListener("input", () => { grosor = Number(inpGrosor.value); });

// Guarda el dibujo como PNG. Se arma un canvas nuevo, se voltea en espejo (para que quede igual
// a lo que se ve en pantalla) y se pinta encima del video o de un fondo oscuro.
function descargarDibujo(conCamara) {
  const c = document.createElement("canvas");
  c.width = dibujo.width;
  c.height = dibujo.height;
  const g = c.getContext("2d");
  g.translate(c.width, 0);
  g.scale(-1, 1);                        // espejo: igual a lo que se ve en pantalla
  if (conCamara) g.drawImage(video, 0, 0, c.width, c.height);
  else { g.fillStyle = "#202020"; g.fillRect(0, 0, c.width, c.height); }   // fondo oscuro: el trazo blanco se ve
  g.drawImage(dibujo, 0, 0);
  bajar(c.toDataURL("image/png"), "dibujo.png");
  registrar("Dibujo descargado");
}
btnPng.addEventListener("click", () => descargarDibujo(false));
btnPngCam.addEventListener("click", () => descargarDibujo(true));

// 11d) MODO CLÁSICO (lo que hacía el programa de Python) ---------------------------
// MODO CLÁSICO: es lo que hacía el programa de Python (captura cada 2 s, hasta 10 fotos, sale con Q).
function detenerClasico(mensaje) {
  clasico.activo = false;
  btnClasico.textContent = "Iniciar captura";
  txtClasico.textContent = `${mensaje ?? "Detenido."} Fotos: ${clasico.n}/${MAX_CLASICO}`;
  registrar(mensaje ?? "Captura clásica detenida");
}
btnClasico.addEventListener("click", () => {
  if (clasico.activo) return detenerClasico("Detenido con el botón.");
  clasico.activo = true;
  clasico.n = 0;
  btnClasico.textContent = "Detener captura";
  txtClasico.textContent = `Capturando... Fotos: 0/${MAX_CLASICO}`;
});
document.addEventListener("keydown", (e) => {          // 'q' para salir, como en Python
  if (e.key.toLowerCase() === "q" && clasico.activo) detenerClasico("Saliendo (tecla Q).");
});
// Se ejecuta en cada cuadro, pero solo actúa si ya pasaron 2 s desde la última captura y hay
// rostros a la vista. Se detiene sola al llegar a MAX_CLASICO (10) fotos.
function procesarClasico() {
  if (!clasico.activo || cajasRostro.length === 0) return;
  const ahora = performance.now();
  if (ahora - clasico.ultimo < ESPERA_FOTO_MS) return;   // 2 s entre capturas
  clasico.ultimo = ahora;
  for (const [i, caja] of cajasRostro.entries()) {
    if (clasico.n >= MAX_CLASICO) break;
    guardarFoto(caja, i);
    clasico.n++;
  }
  txtClasico.textContent = `Capturando... Fotos: ${clasico.n}/${MAX_CLASICO}`;
  if (clasico.n >= MAX_CLASICO) detenerClasico("Se guardaron 10 fotos. Saliendo del programa.");
}

// 11e) QUIZ DE DEDOS -------------------------------------------------------------------
// La respuesta siempre es de 0 a 5 para poder mostrarla con UNA mano.
// El punto se lo lleva la persona cuyo rostro esté más cerca (en horizontal) de la mano correcta.
const azar = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
// QUIZ: genera una suma o resta cuyo resultado siempre está entre 0 y 5, para poder responder
// con los dedos de UNA mano.
function nuevaPregunta() {
  const r = azar(0, 5);
  let texto;
  if (Math.random() < 0.5) { const x = azar(0, r); texto = `${x} + ${r - x}`; }
  else { const y = azar(0, 5 - r); texto = `${r + y} - ${y}`; }
  quiz.resp = r;
  quiz.frames = 0;
  txtPregunta.textContent = `${texto} = ?`;
}
btnPregunta.addEventListener("click", nuevaPregunta);
// Revisa si alguna mano muestra la respuesta correcta durante unos cuadros seguidos. El punto se
// asigna a la persona cuyo rostro está más cerca (en horizontal) de esa mano: es una aproximación.
// Tras acertar hay una pausa de 2 s para no contar el mismo acierto varias veces.
function procesarQuiz(manos) {
  if (quiz.resp < 0) nuevaPregunta();
  if (performance.now() < quiz.bloqueo) return;           // pausa corta tras un acierto
  const m = manos.find(h => h.n === quiz.resp);
  quiz.frames = m ? quiz.frames + 1 : 0;
  if (quiz.frames < FRAMES_ESTABLES + 2) return;          // la respuesta debe mantenerse un momento

  let quien = "Alguien", mejor = Infinity;
  cajasRostro.forEach((c, i) => {
    const d = Math.abs(c.x + c.ancho / 2 - m.puntos[0].x * lienzo.width);
    if (d < mejor) { mejor = d; quien = `Persona ${i + 1}`; }
  });
  quiz.puntos[quien] = (quiz.puntos[quien] || 0) + 1;
  txtPuntos.textContent = "Puntos: " + Object.entries(quiz.puntos).map(([k, v]) => `${k} ${v}`).join(" - ");
  txtPregunta.textContent = `Correcto, ${quien}!`;
  registrar(`Quiz: acertó ${quien}`);
  quiz.bloqueo = performance.now() + 2000;
  setTimeout(nuevaPregunta, 2000);
}

// 11f) SELECTOR DE MODO y extras de pantalla ----------------------------------------
const TEXTOS_MODO = {
  gestos: TEXTO_INICIAL,
  dibujo: "Modo dibujo: levanta solo el índice para dibujar.",
  clasico: "Captura clásica: una foto cada 2 s mientras haya rostros. Sale con Q.",
  quiz: "Quiz: muestra con los dedos la respuesta (0 a 5).",
  vigilancia: "Vigilancia: arma el modo; suena una alarma si alguien aparece.",
  sonrisa: "Sonrisa: sonríe unos instantes y se toma la foto.",
  ppt: "Piedra (puño), papel (mano abierta) o tijera (índice + medio).",
  presentacion: "Presentación: 1 dedo = siguiente, 2 = anterior, 5 = inicio."
};
// Al cambiar de modo: se reinician los lápices y el gesto confirmado, se muestran solo los
// controles de ese modo (los elementos con data-modo), y se apagan funciones del modo anterior.
selModo.addEventListener("change", () => {
  modo = selModo.value;
  modoDibujo = modo === "dibujo";
  lapices = [];
  conteoConfirmado = -1;
  document.querySelectorAll("[data-modo]").forEach(s => { s.hidden = s.dataset.modo !== modo; });
  if (modo !== "clasico" && clasico.activo) detenerClasico("Detenido al cambiar de modo.");
  if (modo === "quiz") nuevaPregunta();
  if (modo !== "vigilancia" && vig.armado) desarmar();
  if (modo === "sonrisa") crearLandmarker();
  if (modo === "presentacion") mostrarDiapo();
  accionTexto.textContent = TEXTOS_MODO[modo];
  registrar(`Modo: ${selModo.selectedOptions[0].textContent}`);
});

// Pantalla completa. Hay tres caminos porque los navegadores no son iguales: la API estándar,
// la versión con prefijo webkit (Safari) y, si ninguna funciona (iPhone), una clase CSS que
// agranda el visor a toda la ventana.
btnPantalla.addEventListener("click", () => {
  const caja = document.querySelector(".escenario");
  if (document.fullscreenElement || document.webkitFullscreenElement) {
    if (document.exitFullscreen) document.exitFullscreen();
    else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
  } else if (caja.classList.contains("is-fullscreen")) {
    caja.classList.remove("is-fullscreen");
  } else {
    const req = caja.requestFullscreen || caja.webkitRequestFullscreen;
    if (req) {
      req.call(caja).catch(() => caja.classList.add("is-fullscreen"));
    } else {
      // Fallback para iPhone / navegadores móviles sin Fullscreen API en divs
      caja.classList.add("is-fullscreen");
    }
  }
});

// En celulares, salir de pantalla completa tocando el visor o tecla Escape
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") document.querySelector(".escenario")?.classList.remove("is-fullscreen");
});

// 11g) VIGILANCIA ------------------------------------------------------------------
// Alerta = aparece un rostro después de ESPERA_VACIO_MS segundos sin nadie.
const vig = { armado: false, ultimoVisto: 0, bloqueo: 0 };
let ctxAudio = null;
// VIGILANCIA: la alarma se genera por código con un oscilador de audio (sin archivos de sonido).
// El navegador solo permite audio después de un clic del usuario, por eso se crea al armar.
function pitar() {                       // 3 pitidos con un oscilador (sonido generado por código, sin archivos)
  if (!ctxAudio) return;
  [0, 0.25, 0.5].forEach(t => {
    const o = ctxAudio.createOscillator(), g = ctxAudio.createGain();
    o.type = "square"; o.frequency.value = 880; g.gain.value = 0.15;
    o.connect(g); g.connect(ctxAudio.destination);
    o.start(ctxAudio.currentTime + t); o.stop(ctxAudio.currentTime + t + 0.15);
  });
}
function desarmar() {
  vig.armado = false;
  btnVig.textContent = "Armar vigilancia";
  txtVig.textContent = "Desarmada.";
  document.documentElement.style.setProperty("--acento", "#c0222a");
  registrar("Vigilancia desarmada");
}
btnVig.addEventListener("click", () => {
  if (vig.armado) return desarmar();
  ctxAudio ??= new AudioContext();       // el navegador solo permite sonido después de un clic
  vig.armado = true;
  vig.bloqueo = 0;
  vig.ultimoVisto = performance.now() - (cajasRostro.length ? 0 : 10000);   // si ya hay alguien, no alerta al armar
  btnVig.textContent = "Desarmar vigilancia";
  txtVig.textContent = "Armada: vigilando...";
  registrar("Vigilancia armada");
});
// Hay alerta cuando aparece un rostro DESPUÉS de ESPERA_VACIO_MS (3 s) sin nadie. "ultimoVisto"
// guarda la última vez que se vio a alguien; si la diferencia es grande, alguien acaba de llegar.
// Tras una alerta hay 5 s de bloqueo para que no suene sin parar.
function procesarVigilancia() {
  if (!vig.armado || cajasRostro.length === 0) return;
  const ahora = performance.now();
  if (ahora - vig.ultimoVisto > ESPERA_VACIO_MS && ahora > vig.bloqueo) {
    pitar();
    cajasRostro.forEach(guardarFoto);    // foto de quien apareció
    registrar("ALERTA: apareció alguien");
    txtVig.textContent = "ALERTA: se detectó a alguien.";
    document.documentElement.style.setProperty("--acento", "#ff0000");
    vig.bloqueo = ahora + 5000;          // 5 s sin repetir la alerta
    setTimeout(() => { if (vig.armado) { txtVig.textContent = "Armada: vigilando..."; document.documentElement.style.setProperty("--acento", "#c0222a"); } }, 3000);
  }
  vig.ultimoVisto = ahora;
}

// 11h) SONRISA -----------------------------------------------------------------------
// FaceLandmarker da "blendshapes": números de 0 a 1 por expresión (mouthSmileLeft/Right = sonrisa).
// Se carga solo al entrar a este modo, para no gastar recursos en los demás.
let landmarker = null, cargandoLM = false;
const smile = { frames: [], ultimo: 0 };
// SONRISA: FaceLandmarker entrega 478 puntos del rostro y "blendshapes", números de 0 a 1 que
// miden cada expresión. Se carga solo al entrar a este modo para no gastar recursos en los demás.
async function crearLandmarker() {
  if (landmarker || cargandoLM) return;
  cargandoLM = true;
  txtSonrisa.textContent = "Cargando modelo de sonrisa...";
  const crear = (delegate) => FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODELO_LANDMARKS, delegate },
    runningMode: "VIDEO", numFaces: 4, outputFaceBlendshapes: true
  });
  try { landmarker = await crear("GPU"); } catch { landmarker = await crear("CPU"); }
  txtSonrisa.textContent = "Sonríe para salir en la foto.";
}
// Calcula la sonrisa como el promedio de mouthSmileLeft y mouthSmileRight. Si pasa de 0.5 durante
// 8 cuadros seguidos (para ignorar gestos pasajeros), toma la foto de esa persona.
function procesarSonrisa() {
  if (!landmarker) return;
  const ahora = performance.now();
  const res = landmarker.detectForVideo(video, ahora);
  res.faceLandmarks.forEach((pts, i) => {
    const cat = res.faceBlendshapes[i].categories;
    const val = (n) => cat.find(c => c.categoryName === n)?.score ?? 0;
    const sonrisa = (val("mouthSmileLeft") + val("mouthSmileRight")) / 2;
    const xs = pts.map(p => p.x * lienzo.width), ys = pts.map(p => p.y * lienzo.height);
    const x1 = Math.max(Math.min(...xs) - PADDING, 0), y1 = Math.max(Math.min(...ys) - PADDING, 0);
    const x2 = Math.min(Math.max(...xs) + PADDING, lienzo.width), y2 = Math.min(Math.max(...ys) + PADDING, lienzo.height);
    const caja = { x: x1, y: y1, ancho: x2 - x1, alto: y2 - y1 };
    ctx.strokeStyle = sonrisa > 0.5 ? "#5cff8a" : "#ff3b3b";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    dibujarEtiqueta(`${Math.round(sonrisa * 100)}%`, (x1 + x2) / 2, y2 + 28, "#7a0e12");
    smile.frames[i] = sonrisa > 0.5 ? (smile.frames[i] || 0) + 1 : 0;   // debe mantenerse ~8 frames
    if (smile.frames[i] >= 8 && ahora - smile.ultimo > ESPERA_FOTO_MS) {
      smile.ultimo = ahora; smile.frames[i] = 0;
      guardarFoto(caja, i);
    }
  });
}

// 11i) PIEDRA, PAPEL O TIJERA ------------------------------------------------------------
let ultimasManos = [];
const ppt = { tu: 0, pc: 0, ocupado: false };
const JUGADAS = ["piedra", "papel", "tijera"];
const GANA_A = { piedra: "tijera", papel: "piedra", tijera: "papel" };   // clave le gana a valor
// PIEDRA, PAPEL O TIJERA: puño = piedra, mano abierta = papel, índice + medio = tijera.
function jugadaDe(m) {
  if (!m) return null;
  const [, i, me, an, mi] = m.estirados;
  if (m.n === 0) return "piedra";
  if (m.n === 5) return "papel";
  if (i && me && !an && !mi) return "tijera";
  return null;
}
// Cuenta regresiva de 3 s con setInterval (se ejecuta cada segundo). Al llegar a 0 se lee la mano
// del último cuadro, la computadora elige al azar y GANA_A decide quién gana.
btnPpt.addEventListener("click", () => {
  if (ppt.ocupado) return;
  ppt.ocupado = true;
  let c = 3;
  txtPpt.textContent = c;
  const t = setInterval(() => {          // cuenta regresiva 3, 2, 1 y se lee tu mano
    c--;
    if (c > 0) { txtPpt.textContent = c; return; }
    clearInterval(t);
    ppt.ocupado = false;
    const tuJugada = jugadaDe(ultimasManos[0]);
    if (!tuJugada) { txtPpt.textContent = "No te vi bien. Repite."; return; }
    const pc = JUGADAS[azar(0, 2)];
    let r = "Empate";
    if (GANA_A[tuJugada] === pc) { ppt.tu++; r = "Ganaste"; }
    else if (GANA_A[pc] === tuJugada) { ppt.pc++; r = "Perdiste"; }
    txtPpt.textContent = `${tuJugada} vs ${pc}: ${r}`;
    txtPptPuntos.textContent = `Tú ${ppt.tu} - Compu ${ppt.pc}`;
    registrar(`Piedra-papel-tijera: ${tuJugada} vs ${pc} (${r})`);
  }, 1000);
});

// 11j) PRESENTACIÓN CON GESTOS ----------------------------------------------------------
const DIAPOSITIVAS = [   // edita estos textos con los de tu defensa
  { t: "Reconocimiento de rostro y manos", x: "Clase III - INFRAMEN" },
  { t: "Librería", x: "MediaPipe Tasks Vision: rostros y 21 puntos por mano" },
  { t: "Contar dedos", x: "Estirado = la punta está más lejos de la muñeca que su articulación" },
  { t: "Modo dibujo", x: "La punta del índice (punto 8) funciona como lápiz" },
  { t: "Gracias", x: "¿Preguntas?" }
];
let diapo = 0, diapoUltimo = 0, diapoFrames = 0, diapoGesto = -1;
function mostrarDiapo() {
  const d = DIAPOSITIVAS[diapo];
  $("diapo").textContent = `${diapo + 1}/${DIAPOSITIVAS.length}\n${d.t}\n${d.x}`;
}
function irDiapo(n) {
  diapo = Math.min(DIAPOSITIVAS.length - 1, Math.max(0, n));
  mostrarDiapo();
}
btnAnt.addEventListener("click", () => irDiapo(diapo - 1));
btnSig.addEventListener("click", () => irDiapo(diapo + 1));
// PRESENTACIÓN: 1 dedo = siguiente, 2 = anterior, 5 = inicio. El gesto debe mantenerse unos cuadros
// y hay una pausa de 1.2 s entre cambios, para no pasar varias diapositivas de golpe.
function procesarPresentacion(manos) {
  const n = manos[0]?.n ?? -1;           // se usa solo la primera mano
  diapoFrames = (n === diapoGesto) ? diapoFrames + 1 : 1;
  diapoGesto = n;
  if (diapoFrames < FRAMES_ESTABLES + 2 || performance.now() - diapoUltimo < 1200) return;   // gesto estable + pausa de 1.2 s
  if (n === 1) irDiapo(diapo + 1);
  else if (n === 2) irDiapo(diapo - 1);
  else if (n === 5) irDiapo(0);
  else return;
  diapoUltimo = performance.now();
}

// 11k) SELECTOR DE CÁMARA -------------------------------------------------------------------
// Llena la lista de cámaras disponibles y marca la que está en uso.
async function llenarCamaras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const lista = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput");
    const actual = video.srcObject?.getVideoTracks?.()[0]?.getSettings?.()?.deviceId ?? "";
    selCamara.innerHTML = "";
    lista.forEach((d, i) => {
      const o = document.createElement("option");
      o.value = d.deviceId;
      o.textContent = d.label || `Cámara ${i + 1}`;
      o.selected = d.deviceId === actual;
      selCamara.appendChild(o);
    });
  } catch (e) {
    console.warn("No se pudo enumerar cámaras:", e);
  }
}
// Si la cámara elegida no abre (otra aplicación la está usando, falta permiso, etc.) mostramos
// el motivo y dejamos funcionando la cámara anterior, en vez de quedar con el video congelado.
selCamara.addEventListener("change", async () => {
  try {
    await encenderCamara(selCamara.value);
  } catch (error) {
    console.error(error);
    estado.textContent = `No se pudo abrir esa cámara (${error.name}). ¿La usa otra aplicación?`;
    await llenarCamaras();   // devuelve el selector a la cámara que sigue activa
  }
});

// 12) BUCLE PRINCIPAL ------------------------------------------------------
// BUCLE PRINCIPAL: se repite solo con requestAnimationFrame, que pide al navegador volver a
// ejecutarlo antes de dibujar el siguiente cuadro (más eficiente que un while infinito, que
// congelaría la página).
// Orden en cada vuelta:
//   1) ¿hay un cuadro de video nuevo? Si no, no se repite el trabajo.
//   2) la IA detecta rostros y manos   3) se limpia el canvas y se dibuja lo detectado
//   4) se actualiza el panel (total de dedos, estado)   5) se llama a la función del modo activo
function bucle() {
  // readyState >= 2 y videoWidth > 0 = el video ya tiene un cuadro con imagen (evita analizar un video vacío)
  if (video.readyState >= 2 && video.videoWidth > 0 && video.currentTime !== ultimoTiempoVideo) {
    ultimoTiempoVideo = video.currentTime;
    const ahora = performance.now();   // MediaPipe exige una marca de tiempo creciente
    fps.n++;                           // contador de FPS: se muestra cada segundo
    if (ahora - fps.t >= 1000) { txtFps.textContent = fps.n; fps.n = 0; fps.t = ahora; }

    const resRostro = detectorRostro.detectForVideo(video, ahora);
    const resMano = detectorMano.detectForVideo(video, ahora);
    ctx.clearRect(0, 0, lienzo.width, lienzo.height);

    // ---- Rostros: un recuadro con margen por persona ----
    cajasRostro = resRostro.detections.map((d, i) => {
      const b = d.boundingBox;
      const x1 = Math.max(b.originX - PADDING, 0);
      const y1 = Math.max(b.originY - PADDING, 0);
      const x2 = Math.min(b.originX + b.width + PADDING, lienzo.width);
      const y2 = Math.min(b.originY + b.height + PADDING, lienzo.height);
      ctx.strokeStyle = "#ff3b3b";
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      dibujarEtiqueta(`Persona ${i + 1}`, (x1 + x2) / 2, y1, "#7a0e12");
      return { x: x1, y: y1, ancho: x2 - x1, alto: y2 - y1 };
    });

    // ---- Manos: dedos de cada una, esqueleto y número sobre la muñeca ----
    const manos = resMano.landmarks.map(puntos => {
      const estirados = dedosEstirados(puntos);
      return { puntos, estirados, n: estirados.filter(Boolean).length };
    });
    manos.forEach(m => {
      if (chkEsqueleto.checked) {     // el esqueleto se puede ocultar con la casilla del panel
        dibujante.drawConnectors(m.puntos, HandLandmarker.HAND_CONNECTIONS, { color: "#ffffff", lineWidth: 3 });
        dibujante.drawLandmarks(m.puntos, { color: "#f5d547", radius: 3 });
      }
      dibujarEtiqueta(String(m.n), m.puntos[0].x * lienzo.width, m.puntos[0].y * lienzo.height + 40, "#2a0a0b");
    });

    // ---- Panel ----
    const total = manos.reduce((suma, m) => suma + m.n, 0);
    numDedos.textContent = manos.length ? total : "-";
    detalleManos.textContent = manos.length
      ? manos.map((m, i) => `Mano ${i + 1}: ${m.n}`).join(" - ")
      : "Sin manos a la vista.";
    const r = cajasRostro.length, h = manos.length;
    // Registro: solo anotamos un cambio de cantidad de rostros si se mantiene ~15 frames (evita spam)
    rostrosEstables = (r === rostrosRegistrados) ? 0 : rostrosEstables + 1;
    if (rostrosEstables >= 15) { rostrosRegistrados = r; rostrosEstables = 0; registrar(`${r} rostro(s) a la vista`); }
    estado.textContent = `${r} ${r === 1 ? "rostro" : "rostros"} - ${h} ${h === 1 ? "mano" : "manos"}`;

    // ---- Qué hacemos según el modo ----
    ultimasManos = manos;   // lo usa piedra-papel-tijera cuando termina la cuenta regresiva
    if (modo === "dibujo") procesarDibujo(manos);
    else if (modo === "clasico") procesarClasico();
    else if (modo === "quiz") procesarQuiz(manos);
    else if (modo === "vigilancia") procesarVigilancia();
    else if (modo === "sonrisa") procesarSonrisa();
    else if (modo === "presentacion") procesarPresentacion(manos);
    else procesarGestos(manos);
  }
  requestAnimationFrame(bucle);
}

// 13) ARRANQUE (la cámara exige un clic del usuario) ------------------------
// ARRANQUE: el navegador exige un clic del usuario para activar la cámara. Todo va dentro de un
// try/catch: si el usuario niega el permiso o abre el archivo sin servidor, se muestra un mensaje
// claro en vez de que la página se rompa en silencio.
btnIniciar.addEventListener("click", async () => {
  btnIniciar.disabled = true;
  try {
    await crearDetectores();
    await encenderCamara();
    btnIniciar.remove();
    bucle();
  } catch (error) {
    console.error(error);
    estado.textContent = "No se pudo iniciar: revisa el permiso de la cámara y que uses un servidor local.";
    btnIniciar.disabled = false;
  }
});

// 14) PESTAÑAS DE MODO: botones grandes que controlan el <select> oculto.
// Así el resto del código (que ya escucha el select) no cambia en nada.
document.querySelectorAll("[data-modo-tab]").forEach(boton => {
  boton.addEventListener("click", () => {
    selModo.value = boton.dataset.modoTab;
    selModo.dispatchEvent(new Event("change"));
    document.querySelectorAll("[data-modo-tab]").forEach(b => b.classList.toggle("activo", b === boton));
  });
});