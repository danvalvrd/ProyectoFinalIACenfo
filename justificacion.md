## Justificación Técnica de Decisiones en la Sección 5: Ingeniería y Selección de Características

### 1. Selección de Variables (Feature Selection)

**Decisión**: Se seleccionaron exclusivamente las variables primarias cuantitativas (mediciones físico-químicas) y se excluyeron identificadores, variables categóricas geográficas (e.g., MUNICIPIO, ACUIFERO) y aquellas de alta cardinalidad.

**Justificación**:
- **Cardinalidad**: El análisis previo mostró que variables como MUNICIPIO y ACUIFERO tienen >200 valores únicos, lo que dificulta la agrupación y puede inducir ruido y sobreajuste.
- **Redundancia**: Identificadores como CLAVE y SITIO son únicos por muestra y no aportan valor predictivo.
- **Geolocalización**: Se retuvieron LONGITUD y LATITUD como variables numéricas para permitir el análisis espacial en etapas posteriores, pero no se incluyeron como input en el clustering base, evitando que la segmentación se vea dominada por la localización geográfica.

### 2. Transformación de Variables (Feature Engineering)

**Decisión**: Se aplicó transformación logarítmica a variables con distribuciones fuertemente sesgadas (e.g., Conductividad, Hierro, Arsénico). Posteriormente, se estandarizaron todas las variables numéricas seleccionadas mediante z-score.

**Justificación**:
- **Log-transform**: Muchas variables presentan colas largas y valores extremos (outliers). La transformación logarítmica reduce el sesgo, mejora la separabilidad de los clusters y estabiliza la varianza.
- **Estandarización (z-score)**: K-Means es sensible a la escala. Sin estandarizar, variables con rangos mayores dominan la distancia euclídea, sesgando la agrupación. El z-score garantiza que cada variable aporte de forma equivalente, cumpliendo buenas prácticas para clustering.

### 3. Manejo de Valores Faltantes

**Decisión**: Se imputaron valores faltantes utilizando la mediana para variables numéricas, siguiendo el procedimiento definido en la Sección 3. Las etiquetas de calidad asociadas se inferieron programáticamente usando las reglas del archivo de metadatos.

**Justificación**:
- **Robustez estadística**: La mediana es menos sensible a outliers y distribuciones sesgadas, preservando la representatividad de la muestra.
- **Consistencia lógica**: Las etiquetas categóricas derivadas de los valores imputados aseguran integridad entre las variables numéricas y sus clasificaciones, evitando inconsistencias que puedan distorsionar el clustering.

### 4. Eliminación de Variables Redundantes

**Decisión**: Se eliminaron variables altamente correlacionadas entre sí tras análisis de matriz de correlación (por ejemplo, Sólidos Disueltos vs. Conductividad).

**Justificación**:
- **Multicolinealidad**: Variables redundantes afectan la interpretación y pueden causar que los clusters representen artefactos matemáticos en vez de patrones reales del agua.
- **Parsimony**: Un modelo más parsimonioso es más interpretable y menos propenso a sobreajuste.

### 5. Construcción del Modelo Base

**Decisión**: El modelo base para clustering se construyó únicamente con las variables seleccionadas, transformadas y estandarizadas, excluyendo ingeniería de ratios geoquímicos o agregados complejos en la primera iteración.

**Justificación**:
- **Validación metodológica**: Un modelo base simple permite validar la capacidad de segmentación de los patrones primarios sin contaminar el análisis con posibles artefactos de ingeniería avanzada.
- **Incrementalidad**: La complejización se realiza post-hoc, permitiendo medir la ganancia real de agregar nuevas variables derivadas.

### 6. Documentación y Reproducibilidad

**Decisión**: Todas las transformaciones fueron documentadas con código reproducible y justificación ejecutiva en el notebook.

**Justificación**:
- **Transparencia**: Permite auditoría y revisión por pares.
- **Reproducibilidad**: Facilita la replicación del análisis y asegura que los resultados sean robustos y confiables.

---

**Conclusión**:  
Las decisiones en ingeniería y selección de características siguieron principios de robustez estadística, interpretabilidad y alineación con estándares en ciencia de datos. Esto asegura que el modelo final de clustering refleje patrones reales y accionables en la calidad del agua, y no artefactos derivados de las variables de entrada o su procesamiento.
