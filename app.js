// =====================================================================
//  app.js  –  Varios rostros + varias manos + contador de dedos + dibujo
//  Librería: MediaPipe Tasks Vision (Google), cargada desde un CDN.
// =====================================================================

// 1) IMPORTAR LA LIBRERÍA ---------------------------------------------
import {
  FilesetResolver, // motor WebAssembly que corre los modelos
  FaceDetector,    // detector de rostros (detecta TODOS los que haya)
  FaceLandmarker,  // puntos del rostro + expresiones (sirve para detectar la sonrisa)
  HandLandmarker,  // detector de manos (21 puntos por mano)
  DrawingUtils     // ayudante para dibujar puntos y líneas
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// 2) CONSTANTES --------------------------------------------------------
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
const MAX_MANOS = 4;         // 2 personas x 2 manos. Si la PC va lenta, bájalo a 4.
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
      numHands: MAX_MANOS     // ANTES era 1: ahora detecta hasta 6 manos a la vez
    });
  };
  try { await crear("GPU"); } catch { await crear("CPU"); }
}

// 6) ENCENDER LA CÁMARA -----------------------------------------------
async function encenderCamara(idCamara) {
  video.srcObject?.getTracks().forEach(t => t.stop());   // apaga la cámara anterior
  // facingMode "user" = cámara frontal en el celular. "ideal" = pide ese tamaño, pero acepta otro.
  const flujo = await navigator.mediaDevices.getUserMedia({
    video: { ...(idCamara ? { deviceId: { exact: idCamara } } : { facingMode: "user" }), width: { ideal: 640 }, height: { ideal: 480 } }
  });
  video.srcObject = flujo;
  await video.play();
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
function aplicarAccion(n) {
  const accion = ACCIONES[n];
  document.documentElement.style.setProperty("--acento", accion.color);
  accionTexto.textContent = accion.texto;
  if (n === 5) tomarFotos();
}

// Con varias manos usamos "consenso": la acción solo se activa si TODAS las manos
// muestran el mismo número. Así dos personas con gestos distintos no se pisan.
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

function borrarDibujo() {
  ctxDibujo.clearRect(0, 0, dibujo.width, dibujo.height);
  lapices = [];
  trazos = [];
}


// 11b) REGISTRO (bitácora) y descargas ---------------------------------------
function registrar(evento) {
  registros.push({ fecha: new Date().toLocaleString("es-SV"), evento });
  const li = document.createElement("li");
  li.textContent = `${new Date().toLocaleTimeString("es-SV")} - ${evento}`;
  listaRegistro.prepend(li);
  if (listaRegistro.children.length > 40) listaRegistro.lastChild.remove();   // solo se muestran los últimos 40
}

function bajar(url, nombre) {            // descarga cualquier URL como archivo
  const enlace = document.createElement("a");
  enlace.href = url;
  enlace.download = nombre;
  enlace.click();
}

btnCsv.addEventListener("click", () => {
  const filas = registros.map(r => `"${r.fecha}","${r.evento.replace(/"/g, '""')}"`);
  // \ufeff = marca para que Excel lea bien las tildes
  const csv = "\ufeffFecha,Evento\n" + filas.join("\n");
  bajar(URL.createObjectURL(new Blob([csv], { type: "text/csv" })), "registro.csv");
});

btnZip.addEventListener("click", async () => {
  if (fotos.length === 0) return;
  const zip = new JSZip();   // JSZip se carga con un <script> en index.html
  fotos.forEach(f => zip.file(f.nombre, f.url.split(",")[1], { base64: true }));
  bajar(URL.createObjectURL(await zip.generateAsync({ type: "blob" })), "fotos.zip");
});

// 11c) DIBUJO: deshacer, color, grosor y descarga --------------------------------
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
function jugadaDe(m) {
  if (!m) return null;
  const [, i, me, an, mi] = m.estirados;
  if (m.n === 0) return "piedra";
  if (m.n === 5) return "papel";
  if (i && me && !an && !mi) return "tijera";
  return null;
}
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
selCamara.addEventListener("change", () => encenderCamara(selCamara.value));

// 12) BUCLE PRINCIPAL ------------------------------------------------------
function bucle() {
  if (video.currentTime !== ultimoTiempoVideo) {
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