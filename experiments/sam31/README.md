# SAM 3.1: backend predeterminado y experimento temporal

Código oficial de Meta en `third_party/SAM31`, commit
`0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`. Pesos `facebook/sam3.1`, revisión
`daa63191845a41281374e725f4c9e51c7a824460`, archivo `sam3.1_multiplex.pt` (3.26 GiB).
SHA256 verificado:
`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.

Python 3.12 y PyTorch 2.7.1 CUDA 12.6 en `.venv` propio. `pose/live_pose.py`
usa SAM 3.1 por defecto para máscaras de inicio y recuperación, mediante un
worker residente. Cada solicitud abre y cierra una sesión de imagen; el tracking
6D continuo sigue a cargo de FoundationPose. La propagación temporal del benchmark
no se ejecuta en el vivo. `--sam-backend sam3` o `--sam-backend efficientsam3`
seleccionan los otros modelos explícitamente. No hay fallback entre modelos.
Durante la búsqueda de recuperación con SAM 3.1 se cede la GPU: termina la
inferencia de profundidad en curso y se pausa el cálculo de profundidad/pose
hasta obtener la máscara. La captura y la ventana siguen actualizándose.
Esto evita la saturación observada con las tres inferencias simultáneas en 8 GiB.
El directorio de código y los pesos están ignorados
por Git. En otra máquina clonar el repositorio oficial en la misma ruta y
seleccionar el commit anterior antes de sincronizar.

```powershell
uv sync --project experiments/sam31
uv run --project experiments/sam31 python experiments/sam31/download_models.py
uv run --project experiments/sam31 python experiments/sam31/benchmark_video.py --out reports/sam31_nuevo --frames 161 --repeats 2
```

Para evaluar entrega causal sin el buffer inicial de 15 frames añadir
`--hotstart-delay 0`. Sigue siendo un replay offline con imágenes precargadas;
esta opción no convierte el predictor en un worker de cámara.

La descarga lee credenciales estándar de Hugging Face o `HF_TOKEN` del `.env`
de la raíz, que también está ignorado. Requiere acceso al modelo concedido a la
cuenta. No imprime ni copia el token.

El benchmark usa los RGB rectificados congelados del replay FS4 (frames 60–2460,
paso 15), guardados sin pérdidas como PNG. Corre la segmentación temporal oficial
con texto `computer mouse`, máximo un objeto, multiplex 16 y batch de detección
1 para reducir memoria. Frames en CPU e inferencia autocast BF16. Por defecto
`--weights-dtype bf16-vision` convierte el tronco visual a BF16; el resto mantiene
FP32 porque el decoder exige FP32 en sus FFN. `--weights-dtype fp32` conserva
todos los pesos en FP32. Convertir el modelo entero a BF16 no es compatible con
esas FFN y se descartó tras comprobar el error de tipos.
No utiliza FlashAttention 3 ni torch.compile. La segunda sesión permite separar
el primer uso de una pasada con kernels ya inicializados.

Las latencias de entrega incluyen el buffering temporal del predictor; no son
tiempos independientes por imagen. El wall total incluye persistir los resultados,
y `delivery_sum_s` excluye esa persistencia. No mide FS, FoundationPose, cámara
ni rendimiento del sistema completo. La IoU comparada con máscaras SAM3 guardadas
es acuerdo entre modelos, no precisión contra ground truth.

Adaptaciones locales sin editar upstream:

- `use_rope_real=False` usa la representación compleja original y evita 64
  buffers derivados de RoPE ausentes del checkpoint. La carga final exige cero
  claves faltantes/inesperadas. El builder oficial intenta primero cargar todo
  el checkpoint en un tracker parcial y muestra advertencias; se auditan ambas
  cargas, pero la final sobre el modelo completo es la que valida los pesos.
- `compat.py` acepta el argumento `offload_state_to_cpu=False` que la API base
  pasa y el modelo multiplex no declara. Si se pide `True`, falla explícitamente.
- El decoder oficial fuerza Flash Attention, que el wheel Windows de PyTorch
  2.7.1 no incluye. `compat.py` permite SDPA Efficient/Math en ese decoder. Se
  conserva la operación de atención y los pesos; los kernels y su rendimiento
  difieren de la configuración optimizada de Meta.

Si ocurre un error se conserva en `summary.json`, junto a configuración y
resultados parciales. Un benchmark fallido no se presenta como ejecutado.

[Resultados medidos el 11 de octubre](../../reports/sam31_experimento_20261011.md).
