# Trainium Frontier — archivo de Team Quanta

Cerramos la campaña activa el **1 de noviembre de 2026**. Publicamos código,
resultados y lecciones de las dos fases, incluyendo los experimentos negativos.

En la Fase 1, la lectura final del tablero que conservamos nos situó terceros con
**0.9471819509 BPB**, usando el paquete 48. En la Fase 2, nuestro mejor envío por
la combinación de métricas fue el 250: **0.7233577472 BPB / 0.2992382166 CORE**.
Otro modelo llegó a **0.7166185501 BPB**, pero perdió CORE. No demostramos el salto
a CORE superior a 0.4. La competición de Fase 2 aún no había terminado al cerrar
nuestra participación activa.

El archivo contiene ocho paquetes de código, dos experimentos de diagnóstico,
mediciones agregadas y un análisis de lo que funcionó y de lo que falló. Las
fuentes copiadas conservan sus bytes originales y un manifiesto SHA-256 permite
verificar su integridad.

- [Resultados y límites de las comparaciones](docs/results.md).
- [Arquitectura y memoria en el host](docs/architecture.md).
- [Qué se necesita para reproducir los experimentos](docs/reproduction.md).
- [Errores de método y lecciones](docs/retrospective.md).
- [Qué se hizo público y qué se conserva privado](docs/publication.md).

Para revisar el archivo sin instalar dependencias ni activar cómputo:

```bash
python3 scripts/verify_archive.py
```

El código histórico requiere el entorno Neuron correspondiente para entrenar.
No incluimos datos, contenido de CORE, pesos, credenciales ni registros internos
de AWS. Las instrucciones no compran capacidad ni levantan instancias.

La lección central: que una implementación compile, sea causal y quepa en memoria
no prueba que mejore CORE. Las señales pequeñas de pruebas cortas necesitan una
confirmación que no conseguimos establecer como predictor fiable del resultado
final. Este archivo documenta esa limitación, sin presentar los fallos de una
receta como prueba de que toda su familia sea imposible.

Código bajo [Apache 2.0](LICENSE), con las atribuciones correspondientes al kit
de AWS y a las dependencias. [Documentación principal en inglés](README.md).
