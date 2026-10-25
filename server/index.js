// =====================================================================
//  servidor/index.js  -  API con Express + MySQL
// =====================================================================
// La página (navegador) NO puede hablar directo con MySQL: sería inseguro. Por eso existe este
// servidor intermedio: la página le manda peticiones HTTP (fetch) y el servidor consulta MySQL.
//   Navegador  --fetch-->  Express (este archivo)  --SQL-->  MySQL

require("dotenv").config();                      // lee el archivo .env (usuario y clave de MySQL, fuera del código)
const express = require("express");
const cors = require("cors");
const mysql = require("mysql2/promise");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const app = express();
const PUERTO = process.env.PORT || 3000;
const CARPETA_FOTOS = path.join(__dirname, "uploads");
fs.mkdirSync(CARPETA_FOTOS, { recursive: true });   // crea uploads/ si no existe

// CORS: el navegador bloquea peticiones entre "sitios" distintos (Live Server :5500 -> servidor :3000).
// Aquí se permite SOLO a páginas locales (localhost / 127.0.0.1), no a cualquier sitio de internet.
app.use(cors({ origin: [/^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/] }));
app.use(express.json({ limit: "10mb" }));           // las fotos llegan como texto (base64): limitamos el tamaño
app.use("/uploads", express.static(CARPETA_FOTOS)); // las fotos guardadas se pueden ver en /uploads/nombre.jpg

// Pool = varias conexiones reutilizables a MySQL (más eficiente que abrir una por petición).
const pool = mysql.createPool({
  host: process.env.DB_HOST || "localhost",
  user: process.env.DB_USER || "root",
  password: process.env.DB_PASSWORD || "",
  database: process.env.DB_NAME || "inframen_ia",
  waitForConnections: true,
  connectionLimit: 5
});

// Ayudante: si una ruta falla, responde error 500 en vez de dejar el servidor colgado.
const ruta = (fn) => (req, res) => fn(req, res).catch((e) => {
  console.error(e);
  res.status(500).json({ error: "Error del servidor" });
});

// Convierte "data:image/jpeg;base64,...." en un archivo. VALIDA el formato (solo jpeg/png) y
// usa un nombre aleatorio: nunca confiamos en un nombre enviado por el cliente (evita escribir donde no debe).
function guardarImagen(dataUrl) {
  const m = /^data:image\/(jpeg|png);base64,([A-Za-z0-9+/=]+)$/.exec(dataUrl || "");
  if (!m) return null;
  const archivo = `${crypto.randomUUID()}.${m[1] === "jpeg" ? "jpg" : "png"}`;
  fs.writeFileSync(path.join(CARPETA_FOTOS, archivo), Buffer.from(m[2], "base64"));
  return archivo;
}
const texto = (v, max) => (typeof v === "string" && v.trim() && v.trim().length <= max) ? v.trim() : null;

// ---- Salud: la página lo usa para saber si el servidor y MySQL responden ----
app.get("/api/salud", ruta(async (req, res) => {
  await pool.query("SELECT 1");
  res.json({ ok: true });
}));

// ---- FOTOS (CRUD) ----
app.post("/api/fotos", ruta(async (req, res) => {
  const nombre = texto(req.body.nombre, 100), archivo = guardarImagen(req.body.imagen);
  if (!nombre || !archivo) return res.status(400).json({ error: "Datos inválidos" });
  // Los "?" son parámetros: MySQL los trata como DATOS, nunca como código (así se evita la inyección SQL).
  const [r] = await pool.query("INSERT INTO fotos (nombre, archivo) VALUES (?, ?)", [nombre, archivo]);
  res.status(201).json({ id: r.insertId });
}));

app.get("/api/fotos", ruta(async (req, res) => {
  const [filas] = await pool.query("SELECT id, nombre, archivo, creado_en FROM fotos ORDER BY id DESC LIMIT 100");
  res.json(filas.map(f => ({ ...f, url: `/uploads/${f.archivo}` })));
}));

app.delete("/api/fotos/:id", ruta(async (req, res) => {
  const [filas] = await pool.query("SELECT archivo FROM fotos WHERE id = ?", [req.params.id]);
  if (!filas.length) return res.status(404).json({ error: "No existe" });
  fs.rmSync(path.join(CARPETA_FOTOS, path.basename(filas[0].archivo)), { force: true });   // basename: solo el nombre, sin rutas
  await pool.query("DELETE FROM fotos WHERE id = ?", [req.params.id]);
  res.json({ ok: true });
}));

// ---- EVENTOS (bitácora) ----
app.post("/api/eventos", ruta(async (req, res) => {
  const evento = texto(req.body.evento, 200);
  if (!evento) return res.status(400).json({ error: "Datos inválidos" });
  await pool.query("INSERT INTO eventos (evento) VALUES (?)", [evento]);
  res.status(201).json({ ok: true });
}));

app.get("/api/eventos", ruta(async (req, res) => {
  const limite = Math.min(Math.max(parseInt(req.query.limite, 10) || 50, 1), 200);   // número entre 1 y 200
  const [filas] = await pool.query("SELECT id, evento, creado_en FROM eventos ORDER BY id DESC LIMIT ?", [limite]);
  res.json(filas);
}));

// ---- ASISTENCIA (nombre escrito a mano + foto del rostro) ----
app.post("/api/asistencia", ruta(async (req, res) => {
  const persona = texto(req.body.persona, 60);
  if (!persona) return res.status(400).json({ error: "Escribe un nombre válido" });
  let fotoId = null;
  if (req.body.imagen) {
    const archivo = guardarImagen(req.body.imagen);
    if (!archivo) return res.status(400).json({ error: "Imagen inválida" });
    const [f] = await pool.query("INSERT INTO fotos (nombre, archivo) VALUES (?, ?)", [`asistencia_${persona}`.slice(0, 100), archivo]);
    fotoId = f.insertId;
  }
  await pool.query("INSERT INTO asistencia (persona, foto_id) VALUES (?, ?)", [persona, fotoId]);
  res.status(201).json({ ok: true });
}));

app.get("/api/asistencia", ruta(async (req, res) => {
  // LEFT JOIN: une cada asistencia con su foto (si no tiene foto, igual aparece, con archivo NULL).
  const [filas] = await pool.query(
    `SELECT a.id, a.persona, a.creado_en, f.archivo
       FROM asistencia a LEFT JOIN fotos f ON f.id = a.foto_id
      ORDER BY a.id DESC LIMIT 50`);
  res.json(filas.map(f => ({ ...f, url: f.archivo ? `/uploads/${f.archivo}` : null })));
}));

app.delete("/api/asistencia/:id", ruta(async (req, res) => {
  await pool.query("DELETE FROM asistencia WHERE id = ?", [req.params.id]);
  res.json({ ok: true });
}));

app.listen(PUERTO, () => console.log(`Servidor listo en http://localhost:${PUERTO}`));