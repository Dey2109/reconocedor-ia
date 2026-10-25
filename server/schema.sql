-- Ejecuta este archivo UNA vez en MySQL (Workbench, phpMyAdmin o consola).
CREATE DATABASE IF NOT EXISTS inframen_ia CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE inframen_ia;

-- Fotos: en la base solo se guarda el NOMBRE del archivo; la imagen vive en la carpeta uploads/
CREATE TABLE IF NOT EXISTS fotos (
  id INT AUTO_INCREMENT PRIMARY KEY,
  nombre VARCHAR(100) NOT NULL,
  archivo VARCHAR(80) NOT NULL,
  creado_en TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Eventos: la bitácora de la página (modos, fotos, alertas...)
CREATE TABLE IF NOT EXISTS eventos (
  id INT AUTO_INCREMENT PRIMARY KEY,
  evento VARCHAR(200) NOT NULL,
  creado_en TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Asistencia: quién se registró y con qué foto (relación con fotos; si se borra la foto, queda NULL)
CREATE TABLE IF NOT EXISTS asistencia (
  id INT AUTO_INCREMENT PRIMARY KEY,
  persona VARCHAR(60) NOT NULL,
  foto_id INT NULL,
  creado_en TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_asistencia_foto FOREIGN KEY (foto_id) REFERENCES fotos(id) ON DELETE SET NULL
);