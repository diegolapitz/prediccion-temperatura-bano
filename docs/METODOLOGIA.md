# Metodología

## Unidad de análisis

Una fila representa un intervalo entre una medición de temperatura inicial y una medición futura de la misma unidad. La variable objetivo es la temperatura futura real. Se conservan únicamente intervalos que cumplen los filtros de calidad y separación temporal definidos en la configuración y en los scripts de preparación.

## Prevención de fuga temporal

- Los cortes de entrenamiento, validación y evaluación usan la fecha de la **medición objetivo**, no un reparto aleatorio de filas.
- Las variables de señales frecuentes se calculan con períodos completos anteriores al objetivo. El período inicial y el período objetivo se excluyen cuando la hora física exacta de medición no está disponible.
- Identificadores, fechas absolutas y la temperatura futura no se usan como variables directas de los primeros modelos.
- La selección de ventanas, variables y parámetros del modelo de resumen térmico se hizo con cortes anteriores a la cohorte de julio de 2026.

## Comparadores y métricas

Se calculan referencias interpretables antes de entrenar modelos: última temperatura, promedio de dos temperaturas y extrapolación simple. El MAE se reporta tanto globalmente como por subgrupos, en particular aumentos, descensos y temperaturas finales altas. Las comparaciones de mejora usan los mismos casos para ambos métodos; la auditoría del modelo final incluye remuestreo agrupado por unidad.

## Desarrollo del modelo

Los scripts `01` a `10` contienen el desarrollo inicial y análisis exploratorios. Los scripts `15` a `17` agregan señales frecuentes mediante ventanas causales. `18` construye resúmenes térmicos adicionales; `19` entrena y evalúa las salidas del modelo final de esa etapa; `20` reproduce métricas y comparaciones. Los experimentos y el piloto posteriores permanecen en el proyecto privado.

## Interpretación

Las métricas publicadas son resultados reales **agregados y retrospectivos**. No permiten reconstruir registros individuales. Julio de 2026 ya había sido observado durante el desarrollo; no se presenta como prueba virgen. Tampoco se afirma aquí validación prospectiva limpia. Reentrenar con otros datos requeriría verificar equivalencia de esquema, calendario y disponibilidad de señales antes de interpretar el resultado.
