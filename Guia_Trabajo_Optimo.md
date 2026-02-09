# Guía de Trabajo Óptimo con Archivos

Esta guía define el **Estándar de Oro** para trabajar conmigo. Me he "auto-abastecido" con las librerías necesarias para procesar todos estos formatos nativamente en tu entorno.

## 📁 Archivos Locales (.pdf, .xlsx, .docx, .pptx, Img)

Ya tengo instaladas las herramientas para leer, editar y crear estos archivos directamente.

| Formato | Herramienta Instalada (Motor) | Qué puedo hacer "Nativamente" |
| :--- | :--- | :--- |
| **Excel (.xlsx)** | `pandas`, `openpyxl` | Leer tablas masivas, analizar datos, crear reportes, filtrar, corregir, gráficos. |
| **PDF (.pdf)** | `pdfplumber`, `pypdfium2` | Extraer texto preciso, leer tablas, extraer imágenes, combinar/dividir PDFs. |
| **Word (.docx)** | `python-docx` | Leer contenido, redactar documentos nuevos, modificar estilos y formatos. |
| **PowerPoint (.pptx)** | `python-pptx` | Leer diapositivas, extraer texto/imágenes, crear presentaciones nuevas. |
| **Imágenes (.jpg, .png)** | `Pillow (PIL)` | Redimensionar, convertir formatos, ediciones básicas. (Para "ver" contenido, simplemente súbela al chat). |

### ✅ La mejor forma de pasármelos:
Simplemente **mueve o copia el archivo a mi carpeta de trabajo** (actualmente `.../scratch/mining_app`) y dime su nombre.
*Ejemplo: "Analiza la planilla Costos.xlsx que puse en la carpeta".*

---

## ☁️ Google Drive / Docs / Sheets (Web)

Como soy una IA segura, no tengo acceso directo a tu cuenta de Google (ni debería tenerlo por privacidad).

### ✅ La forma óptima: **"Descargar y Arrastrar"**
Para que yo pueda procesar la información con mis nuevas herramientas "pandas/python", haz esto:

1.  **Google Sheets:** Archivo -> Descargar -> **Microsoft Excel (.xlsx)**.
2.  **Google Docs:** Archivo -> Descargar -> **Microsoft Word (.docx)**.
3.  **Google Slides:** Archivo -> Descargar -> **Microsoft PowerPoint (.pptx)**.

Luego, coloca ese archivo descargado en mi carpeta. Así puedo usar toda la potencia de `pandas` y Python para trabajar con tus datos.

---

> **Compromiso:** De ahora en adelante, si pones uno de estos archivos en la carpeta, asumiré que puedo leerlo y procesarlo. No más excusas. 🚀
